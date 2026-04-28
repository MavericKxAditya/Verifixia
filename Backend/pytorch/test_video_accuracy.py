#!/usr/bin/env python3
"""
Video-based accuracy test script for deepfake detection
Tests the trained video model directly on video data
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import cv2
import os
import yaml
import argparse
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import random
from pathlib import Path


class VideoDeepfakeDataset(Dataset):
    """Dataset for video-based deepfake detection testing"""
    def __init__(self, root_dir, transform=None, frames_per_video=10):
        self.root_dir = root_dir
        self.transform = transform
        self.frames_per_video = frames_per_video
        self.samples = []

        # Video extensions
        VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv', '.wmv'}

        real_dir = os.path.join(root_dir, 'Real')
        fake_dir = os.path.join(root_dir, 'Fake')

        all_samples = []

        # Load Real Videos (Label 0)
        if os.path.exists(real_dir):
            for video_file in os.listdir(real_dir):
                if Path(video_file).suffix.lower() in VIDEO_EXTENSIONS:
                    all_samples.append((os.path.join(real_dir, video_file), 0, 'real'))
        else:
            print(f"Warning: Real videos directory not found at {real_dir}")

        # Load Deepfake Videos (Label 1)
        if os.path.exists(fake_dir):
            for video_file in os.listdir(fake_dir):
                if Path(video_file).suffix.lower() in VIDEO_EXTENSIONS:
                    all_samples.append((os.path.join(fake_dir, video_file), 1, 'fake'))
        else:
            print(f"Warning: Deepfake videos directory not found at {fake_dir}")

        # Balance the dataset to have equal real and fake samples
        real_samples = [s for s in all_samples if s[1] == 0]
        fake_samples = [s for s in all_samples if s[1] == 1]
        
        min_count = min(len(real_samples), len(fake_samples))
        if min_count == 0:
            print("Warning: No videos found in one or both categories")
            self.samples = []
        else:
            # Take equal number from each
            self.samples = real_samples[:min_count] + fake_samples[:min_count]
            print(f"Balanced dataset: {min_count} real + {min_count} fake = {len(self.samples)} videos")

    def __len__(self):
        return len(self.samples)

    def extract_frames(self, video_path, num_frames=10):
        """Extract evenly-spaced frames from video"""
        frames = []
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Warning: Could not open video {video_path}")
                return None

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if total_frames < num_frames:
                num_frames = max(1, total_frames)

            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
                ret, frame = cap.read()

                if ret:
                    # Convert BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)

            cap.release()

            return frames if frames else None

        except Exception as e:
            print(f"Error extracting frames from {video_path}: {e}")
            return None

    def __getitem__(self, idx):
        video_path, label, label_name = self.samples[idx]
        frames = self.extract_frames(video_path, self.frames_per_video)

        if frames is None:
            # Return empty tensor if extraction failed
            return torch.zeros(self.frames_per_video, 3, 224, 224), torch.tensor(label, dtype=torch.float), video_path, label_name

        # Apply transforms to each frame
        processed_frames = []
        for frame in frames:
            if self.transform:
                frame = self.transform(frame)
            processed_frames.append(frame)

        # Stack frames: [num_frames, channels, height, width]
        video_tensor = torch.stack(processed_frames)

        return video_tensor, torch.tensor(label, dtype=torch.float), video_path, label_name


class VideoDeepfakeDetector(nn.Module):
    """Video-based deepfake detection model"""
    def __init__(self, use_3d=False):
        super(VideoDeepfakeDetector, self).__init__()
        self.use_3d = use_3d

        # Base CNN for frame processing
        self.backbone = models.resnet50(pretrained=True)
        num_features = self.backbone.fc.in_features

        # Remove classification head
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])

        # Temporal modeling
        if use_3d:
            # 3D convolution for temporal modeling
            self.temporal_conv = nn.Conv3d(2048, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.temporal_pool = nn.AdaptiveAvgPool3d(1)
            temporal_features = 512
        else:
            # LSTM for temporal modeling
            self.temporal_lstm = nn.LSTM(input_size=2048, hidden_size=256,
                                        num_layers=2, batch_first=True, dropout=0.5)
            temporal_features = 256

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(temporal_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: [batch_size, num_frames, channels, height, width]
        """
        batch_size, num_frames, c, h, w = x.shape

        # Process each frame through backbone
        x = x.view(batch_size * num_frames, c, h, w)
        features = self.backbone(x)  # [batch_size*num_frames, 2048, 1, 1]
        features = features.view(batch_size * num_frames, -1)  # [batch_size*num_frames, 2048]

        if self.use_3d:
            # Reshape for 3D convolution
            features = features.view(batch_size, num_frames, -1, 1, 1)
            features = features.permute(0, 2, 1, 3, 4)  # [batch_size, 2048, num_frames, 1, 1]
            features = self.temporal_conv(features)
            features = self.temporal_pool(features)
            features = features.view(batch_size, -1)
        else:
            # LSTM processing
            features = features.view(batch_size, num_frames, -1)
            _, (features, _) = self.temporal_lstm(features)
            features = features[-1]  # Take last hidden state

        # Classification
        output = self.classifier(features)
        return output


def test_video_model(config):
    """Test video-based model on video data"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Load configuration
    data_path = config.get('test_data_path', '../../DATA')
    model_path = config.get('model_path', '../../models/video_deepfake.pth')
    batch_size = config.get('batch_size', 4)
    num_frames = config.get('augmentation', {}).get('num_frames_per_video', 10)
    use_3d = config.get('use_3d', False)

    # Data transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Scale(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create test dataset
    test_dataset = VideoDeepfakeDataset(data_path, transform=transform, frames_per_video=num_frames)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    if len(test_dataset) == 0:
        print("No video data found for testing!")
        return

    # Load model
    model = VideoDeepfakeDetector(use_3d=use_3d)

    if os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("✓ Model loaded")
    else:
        print(f"Warning: Model not found at {model_path}")
        print("Using randomly initialized model (results will be random)")
        # Still test with random weights to show the testing framework works

    model.to(device)
    model.eval()

    # Test the model
    print("\n" + "=" * 60)
    print("TESTING VIDEO MODEL")
    print("=" * 60)

    all_preds = []
    all_labels = []
    all_video_paths = []
    all_label_names = []

    with torch.no_grad():
        for frames, labels, video_paths, label_names in tqdm(test_loader, desc="Testing"):
            frames = frames.to(device)
            labels = labels.to(device)

            outputs = model(frames)
            predictions = (outputs > 0.5).float()

            all_preds.extend(predictions.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
            all_video_paths.extend(video_paths)
            all_label_names.extend(label_names)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Print results
    print("\n" + "=" * 60)
    print("VIDEO ACCURACY TEST RESULTS")
    print("=" * 60)
    print(f"Total videos tested: {len(all_labels)}")
    print(f"Accuracy:          {accuracy*100:.2f}%")
    print(f"Precision:         {precision*100:.2f}%")
    print(f"Recall:           {recall*100:.2f}%")
    print(f"F1 Score:         {f1*100:.2f}%")

    print("\nConfusion Matrix:")
    print(f"True Negatives (Real correctly classified): {cm[0][0]}")
    print(f"False Positives (Real misclassified as Fake): {cm[0][1]}")
    print(f"False Negatives (Fake misclassified as Real): {cm[1][0]}")
    print(f"True Positives (Fake correctly classified): {cm[1][1]}")

    # Save results
    results = {
        'total_videos': len(all_labels),
        'accuracy': round(float(accuracy)*100, 2),
        'precision': round(float(precision)*100, 2),
        'recall': round(float(recall)*100, 2),
        'f1_score': round(float(f1)*100, 2),
        'confusion_matrix': {
            'tn': int(cm[0][0]),
            'fp': int(cm[0][1]),
            'fn': int(cm[1][0]),
            'tp': int(cm[1][1])
        },
        'model_path': model_path,
        'test_data_path': data_path
    }

    import json
    with open('video_accuracy_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to video_accuracy_test_results.json")

    # Print some example predictions
    print("\nSample Predictions:")
    for i in range(min(10, len(all_video_paths))):
        pred_label = "Fake" if all_preds[i] > 0.5 else "Real"
        true_label = "Fake" if all_labels[i] > 0.5 else "Real"
        correct = "✓" if (all_preds[i] > 0.5) == (all_labels[i] > 0.5) else "✗"
        video_name = os.path.basename(all_video_paths[i])
        print(f"{correct} {video_name}: Predicted {pred_label}, True {true_label}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test video-based deepfake detector')
    parser.add_argument('--config', type=str, default='config_video.yaml',
                       help='Path to test config YAML file')

    args = parser.parse_args()

    # Load config
    if not os.path.exists(args.config):
        print(f"Config file {args.config} not found, using default settings")
        config = {
            'test_data_path': '../../DATA',
            'model_path': '../../models/video_deepfake.pth',
            'batch_size': 4,
            'augmentation': {'num_frames_per_video': 10},
            'use_3d': False
        }
    else:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

    test_video_model(config)