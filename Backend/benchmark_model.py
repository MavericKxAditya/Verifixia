#!/usr/bin/env python3
"""
Benchmark script to measure model inference speed
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import time
import numpy as np

class DeepfakeDetectorPretrained(nn.Module):
    """Pre-trained ResNet50-based deepfake detection model"""
    def __init__(self):
        super(DeepfakeDetectorPretrained, self).__init__()
        self.backbone = models.resnet50(pretrained=False)
        num_features = self.backbone.fc.in_features
        original_fc = self.backbone.fc
        self.backbone.fc = nn.Sequential(  # type: ignore[assignment]
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )
        del original_fc

    def forward(self, x):
        return self.backbone(x)

def benchmark_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # Load model
    model = DeepfakeDetectorPretrained()
    model_path = "../models/xception_deepfake.pth"
    
    print("Loading model...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("✓ Model loaded\n")

    # Create test image (already 224x224)
    test_image = Image.new('RGB', (224, 224), color=(0, 0, 0))
    
    # Transform
    # noinspection PyTypeChecker
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])  # type: ignore
    
    image_tensor = transform(test_image).unsqueeze(0).to(device)  # type: ignore

    print("=" * 60)
    print("INFERENCE SPEED BENCHMARK")
    print("=" * 60)
    
    # Warm up
    with torch.no_grad():
        for _ in range(5):
            model(image_tensor)
    
    # Single image inference
    print("\n1. SINGLE IMAGE INFERENCE:")
    times = []
    for i in range(10):
        start = time.time()
        with torch.no_grad():
            output = model(image_tensor)
        end = time.time()
        times.append((end - start) * 1000)  # Convert to ms
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    print(f"   Average: {avg_time:.2f} ms")
    print(f"   Std Dev: {std_time:.2f} ms")
    print(f"   Min:     {min(times):.2f} ms")
    print(f"   Max:     {max(times):.2f} ms")
    print(f"   Rate:    {1000/avg_time:.1f} images/sec")
    
    # Batch inference (32 images)
    print("\n2. BATCH INFERENCE (32 images):")
    batch_image = image_tensor.repeat(32, 1, 1, 1)
    times = []
    for i in range(10):
        start = time.time()
        with torch.no_grad():
            output = model(batch_image)
        end = time.time()
        times.append((end - start) * 1000)
    
    avg_time = np.mean(times)
    per_image_time = avg_time / 32
    print(f"   Total:      {avg_time:.2f} ms")
    print(f"   Per image:  {per_image_time:.2f} ms")
    print(f"   Batch rate: {(1000/per_image_time):.1f} images/sec")
    
    # Memory usage
    print("\n3. MODEL STATISTICS:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total Parameters:    {total_params:,}")
    print(f"   Trainable Parameters: {trainable_params:,}")
    
    # Model size
    import os
    model_size = os.path.getsize(model_path) / (1024 * 1024)
    print(f"   Model File Size:     {model_size:.2f} MB")
    
    print("\n" + "=" * 60)
    print("ResNet50 is efficient and optimized.")
    print("Training time was ~2-3 minutes on CPU with 800 images.")
    print("=" * 60)

if __name__ == '__main__':
    benchmark_model()
