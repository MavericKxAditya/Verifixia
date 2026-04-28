import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List

import cv2
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

try:
    from torchvision.models import efficientnet_b0
except ImportError:
    try:
        from torchvision.models.efficientnet import efficientnet_b0
    except ImportError as exc:
        raise ImportError(
            "DeepFakeTester requires a torchvision build that provides EfficientNet-B0. "
            "Use the Python environment where torch/torchvision are installed together."
        ) from exc


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}
SUPPORTED_EXTENSIONS = VIDEO_EXTENSIONS | IMAGE_EXTENSIONS

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
MODEL_PATH = BASE_DIR.parent / "Notebooks" / "deepfake_cnn_lstm.pth"
FRAMES_PER_VIDEO = 8
IMAGE_SIZE = 224
CLASS_NAMES = ["Real", "Fake"]


def replace_activations_with_relu(module: nn.Module) -> None:
    for name, child in module.named_children():
        if isinstance(child, nn.SiLU):
            setattr(module, name, nn.ReLU(inplace=True))
        else:
            replace_activations_with_relu(child)


class DeepfakeCNNLSTM(nn.Module):
    def __init__(self, dataset_type: str = "video", num_classes: int = 2, frames_per_video: int = 8):
        super().__init__()
        self.dataset_type = dataset_type
        self.frames_per_video = frames_per_video

        self.backbone = efficientnet_b0(weights=None)
        self.backbone.classifier = nn.Identity()
        replace_activations_with_relu(self.backbone)

        self.lstm = nn.LSTM(
            input_size=1280,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dataset_type == "video":
            batch_size, frames, channels, height, width = x.shape
            x = x.view(batch_size * frames, channels, height, width)
            features = self.backbone(x)
            features = features.view(batch_size, frames, 1280)
            _, (hidden, _) = self.lstm(features)
            return self.classifier(hidden[-1])

        features = self.backbone(x)
        return self.classifier(features)


@dataclass
class PredictionResult:
    filename: str
    path: str
    media_type: str
    prediction: str
    predicted_index: int
    confidence: float
    probabilities: dict
    processed_at: str


def build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def load_video_model(device: torch.device) -> nn.Module:
    model = DeepfakeCNNLSTM(dataset_type="video", frames_per_video=FRAMES_PER_VIDEO)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def extract_video_frames(video_path: Path, frames_per_video: int) -> List[Image.Image]:
    cap = cv2.VideoCapture(str(video_path))
    frames: List[Image.Image] = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        dummy = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE))
        return [dummy] * frames_per_video

    indices = [int(i) for i in torch.linspace(0, total_frames - 1, frames_per_video).long()]
    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if ok:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
        else:
            frames.append(frames[-1] if frames else Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE)))

    cap.release()

    while len(frames) < frames_per_video:
        frames.append(frames[-1] if frames else Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE)))

    return frames[:frames_per_video]


def classify_logits(logits: torch.Tensor) -> tuple[str, int, float, dict]:
    probabilities_tensor = torch.softmax(logits, dim=1)[0].detach().cpu()
    predicted_index = int(torch.argmax(probabilities_tensor).item())
    prediction = CLASS_NAMES[predicted_index]
    confidence = float(probabilities_tensor[predicted_index].item() * 100.0)
    probabilities = {
        CLASS_NAMES[index]: round(float(score.item()) * 100.0, 4)
        for index, score in enumerate(probabilities_tensor)
    }
    return prediction, predicted_index, confidence, probabilities


def predict_image(image_path: Path, model: nn.Module, transform: transforms.Compose, device: torch.device) -> PredictionResult:
    image = Image.open(image_path).convert("RGB")
    frame_tensor = transform(image)
    tensor = torch.stack([frame_tensor] * FRAMES_PER_VIDEO).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)

    prediction, predicted_index, confidence, probabilities = classify_logits(logits)
    return PredictionResult(
        filename=image_path.name,
        path=str(image_path),
        media_type="image",
        prediction=prediction,
        predicted_index=predicted_index,
        confidence=round(confidence, 2),
        probabilities=probabilities,
        processed_at=datetime.now().isoformat(),
    )


def predict_video(video_path: Path, model: nn.Module, transform: transforms.Compose, device: torch.device) -> PredictionResult:
    frames = extract_video_frames(video_path, FRAMES_PER_VIDEO)
    tensor = torch.stack([transform(frame) for frame in frames]).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)

    prediction, predicted_index, confidence, probabilities = classify_logits(logits)
    return PredictionResult(
        filename=video_path.name,
        path=str(video_path),
        media_type="video",
        prediction=prediction,
        predicted_index=predicted_index,
        confidence=round(confidence, 2),
        probabilities=probabilities,
        processed_at=datetime.now().isoformat(),
    )


def ensure_paths() -> None:
    INPUT_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)


def find_inputs() -> List[Path]:
    return sorted(
        path for path in INPUT_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    ensure_paths()

    if not MODEL_PATH.exists():
        print(f"Model file not found: {MODEL_PATH}")
        return 1

    media_files = find_inputs()
    if not media_files:
        print(f"No supported files found in {INPUT_DIR}")
        print("Add an image or video file there, then rerun this script.")
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = build_transform()
    video_model = load_video_model(device)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = {
        "run_id": run_id,
        "model_path": str(MODEL_PATH),
        "device": str(device),
        "frames_per_video": FRAMES_PER_VIDEO,
        "image_size": IMAGE_SIZE,
        "input_dir": str(INPUT_DIR),
        "output_dir": str(OUTPUT_DIR),
        "total_files": len(media_files),
        "results": [],
    }

    print(f"Loaded model from {MODEL_PATH}")
    print(f"Found {len(media_files)} file(s) in {INPUT_DIR}")

    for media_path in media_files:
        if media_path.suffix.lower() in VIDEO_EXTENSIONS:
            result = predict_video(media_path, video_model, transform, device)
        else:
            result = predict_image(media_path, video_model, transform, device)

        result_payload = result.__dict__
        summary["results"].append(result_payload)

        output_path = OUTPUT_DIR / f"{media_path.stem}_result.json"
        save_json(output_path, result_payload)
        print(f"{media_path.name}: {result.prediction} ({result.confidence:.2f}%)")

    summary_path = OUTPUT_DIR / f"summary_{run_id}.json"
    save_json(summary_path, summary)
    print(f"Summary saved to {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
