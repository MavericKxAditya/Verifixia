import json
from dataclasses import asdict, dataclass
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
            "This app requires a torchvision build that provides EfficientNet-B0."
        ) from exc


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}
SUPPORTED_EXTENSIONS = VIDEO_EXTENSIONS | IMAGE_EXTENSIONS
CLASS_NAMES = ["Real", "Fake"]
IMAGE_SIZE = 224
FRAMES_PER_VIDEO = 8

APP_DIR = Path(__file__).resolve().parent
INPUT_DIR = APP_DIR / "input"
OUTPUT_DIR = APP_DIR / "outputs"
MODEL_PATH = APP_DIR.parent / "Notebooks" / "deepfake_cnn_lstm.pth"


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
        batch_size, frames, channels, height, width = x.shape
        x = x.view(batch_size * frames, channels, height, width)
        features = self.backbone(x)
        features = features.view(batch_size, frames, 1280)
        _, (hidden, _) = self.lstm(features)
        return self.classifier(hidden[-1])


@dataclass
class PredictionResult:
    filename: str
    path: str
    media_type: str
    prediction: str
    confidence: float
    probabilities: dict
    processed_at: str
    output_file: str


def ensure_app_dirs() -> None:
    INPUT_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)


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


def extract_video_frames(video_path: Path, frames_per_video: int) -> List[Image.Image]:
    cap = cv2.VideoCapture(str(video_path))
    frames: List[Image.Image] = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        fallback = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE))
        return [fallback] * frames_per_video

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


def classify_logits(logits: torch.Tensor) -> tuple[str, float, dict]:
    probabilities_tensor = torch.softmax(logits, dim=1)[0].detach().cpu()
    predicted_index = int(torch.argmax(probabilities_tensor).item())
    prediction = CLASS_NAMES[predicted_index]
    confidence = round(float(probabilities_tensor[predicted_index].item() * 100.0), 2)
    probabilities = {
        CLASS_NAMES[index]: round(float(score.item()) * 100.0, 4)
        for index, score in enumerate(probabilities_tensor)
    }
    return prediction, confidence, probabilities


class DeepfakePredictor:
    def __init__(self) -> None:
        ensure_app_dirs()
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

        self.model_path = MODEL_PATH
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = build_transform()
        self.model = self._load_model()

    def _load_model(self) -> nn.Module:
        model = DeepfakeCNNLSTM(dataset_type="video", frames_per_video=FRAMES_PER_VIDEO)
        state_dict = torch.load(MODEL_PATH, map_location=self.device)
        model.load_state_dict(state_dict, strict=True)
        model.to(self.device)
        model.eval()
        return model

    def _predict_image(self, image_path: Path) -> tuple[str, float, dict]:
        image = Image.open(image_path).convert("RGB")
        frame_tensor = self.transform(image)
        tensor = torch.stack([frame_tensor] * FRAMES_PER_VIDEO).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
        return classify_logits(logits)

    def _predict_video(self, video_path: Path) -> tuple[str, float, dict]:
        frames = extract_video_frames(video_path, FRAMES_PER_VIDEO)
        tensor = torch.stack([self.transform(frame) for frame in frames]).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
        return classify_logits(logits)

    def predict_file(self, file_path: Path) -> PredictionResult:
        suffix = file_path.suffix.lower()
        media_type = "video" if suffix in VIDEO_EXTENSIONS else "image"
        prediction_fn = self._predict_video if media_type == "video" else self._predict_image
        prediction, confidence, probabilities = prediction_fn(file_path)

        output_file = OUTPUT_DIR / f"{file_path.stem}_result.json"
        result = PredictionResult(
            filename=file_path.name,
            path=str(file_path),
            media_type=media_type,
            prediction=prediction,
            confidence=confidence,
            probabilities=probabilities,
            processed_at=datetime.now().isoformat(),
            output_file=str(output_file),
        )
        output_file.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
        return result


def is_supported_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in SUPPORTED_EXTENSIONS
