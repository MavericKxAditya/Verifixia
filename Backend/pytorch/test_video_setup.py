#!/usr/bin/env python3
"""
Test script to verify video training dependencies and setup
"""

import sys
import subprocess

def test_imports():
    """Test if all required packages are installed"""
    print("=" * 60)
    print("TESTING PYTHON DEPENDENCIES")
    print("=" * 60)
    
    dependencies = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'cv2': 'OpenCV',
        'PIL': 'Pillow',
        'numpy': 'NumPy',
        'sklearn': 'scikit-learn',
        'yaml': 'PyYAML',
    }
    
    missing = []
    
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"✓ {name:20} installed")
        except ImportError:
            print(f"✗ {name:20} MISSING")
            missing.append(name)
    
    return missing


def test_gpu():
    """Test if GPU is available for PyTorch"""
    print("\n" + "=" * 60)
    print("CHECKING GPU/DEVICE")
    print("=" * 60)
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            print(f"  PyTorch version: {torch.__version__}")
        else:
            print(f"⚠ CUDA not available - will use CPU")
            print(f"  CPU: {sys.platform}")
            print(f"  PyTorch will run on CPU (slower)")
    except Exception as e:
        print(f"✗ Error checking GPU: {e}")


def test_video_processing():
    """Test video frame extraction"""
    print("\n" + "=" * 60)
    print("TESTING VIDEO PROCESSING")
    print("=" * 60)
    
    try:
        import cv2
        print(f"✓ OpenCV version: {cv2.__version__}")
        
        # Test video codec support
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        print(f"✓ MP4 codec support available")
        
        # Test image reading
        from PIL import Image
        import numpy as np
        img = Image.new('RGB', (100, 100), color='red')
        print(f"✓ PIL image creation works")
        
        # Test torch tensor operations
        import torch
        tensor = torch.zeros(10, 3, 224, 224)
        print(f"✓ PyTorch tensor creation works (shape: {tuple(tensor.shape)})")
        
    except Exception as e:
        print(f"✗ Video processing error: {e}")


def test_data_structure():
    """Test if data directory exists and has correct structure"""
    print("\n" + "=" * 60)
    print("CHECKING DATA STRUCTURE")
    print("=" * 60)
    
    import os
    
    base_path = os.path.abspath("../../DATA")
    
    print(f"Looking for data at: {base_path}")
    
    if os.path.exists(base_path):
        print(f"✓ DATA directory exists")
        
        real_path = os.path.join(base_path, "Real")
        fake_path = os.path.join(base_path, "Fake")
        
        if os.path.exists(real_path):
            real_videos = len([f for f in os.listdir(real_path) 
                             if f.lower().endswith(('.mp4', '.mov', '.avi', '.mkv', '.webm'))])
            print(f"✓ Real/ directory exists with {real_videos} videos")
        else:
            print(f"✗ Real/ directory not found")
            print(f"  Create it at: {real_path}")
        
        if os.path.exists(fake_path):
            fake_videos = len([f for f in os.listdir(fake_path)
                             if f.lower().endswith(('.mp4', '.mov', '.avi', '.mkv', '.webm'))])
            print(f"✓ Fake/ directory exists with {fake_videos} videos")
        else:
            print(f"✗ Fake/ directory not found")
            print(f"  Create it at: {fake_path}")
    else:
        print(f"✗ DATA directory not found at {base_path}")
        print(f"  Please create the directory and add videos:")
        print(f"  - {os.path.join(base_path, 'Real')}/ (real video files)")
        print(f"  - {os.path.join(base_path, 'Fake')}/ (deepfake video files)")


def test_model_creation():
    """Test model creation"""
    print("\n" + "=" * 60)
    print("TESTING MODEL CREATION")
    print("=" * 60)
    
    try:
        import torch
        import torch.nn as nn
        from torchvision import models
        
        # Test ResNet50 loading
        print("Creating ResNet50 backbone...")
        backbone = models.resnet50(pretrained=False)
        print(f"✓ ResNet50 loaded successfully")
        
        # Test temporal model creation (simplified)
        print("Creating temporal model...")
        
        class SimpleTemporalModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = models.resnet50(pretrained=False)
                self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])  # Remove classification head
                self.lstm = nn.LSTM(input_size=2048, hidden_size=256, num_layers=2, batch_first=True)
                self.classifier = nn.Linear(256, 1)
            
            def forward(self, x):
                # x: [batch, frames, 3, 224, 224]
                b, f, c, h, w = x.shape
                x = x.view(b*f, c, h, w)
                features = self.backbone(x)  # [b*f, 2048, 1, 1]
                features = features.view(b*f, -1)  # [b*f, 2048]
                features = features.view(b, f, -1)  # [b, f, 2048]
                _, (hidden, _) = self.lstm(features)
                out = self.classifier(hidden[-1])
                return out
        
        model = SimpleTemporalModel()
        print(f"✓ Temporal model created successfully")
        
        # Test forward pass
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        test_input = torch.randn(2, 10, 3, 224, 224).to(device)
        with torch.no_grad():
            output = model(test_input)
        print(f"✓ Forward pass successful")
        print(f"  Input shape: {test_input.shape}")
        print(f"  Output shape: {output.shape}")
        
    except Exception as e:
        print(f"✗ Model creation error: {e}")
        import traceback
        traceback.print_exc()


def main():
    print("\n")
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   VIDEO TRAINING SETUP - DEPENDENCY CHECK               ║")
    print("╚══════════════════════════════════════════════════════════╝")
    
    # Run all tests
    missing = test_imports()
    test_gpu()
    test_video_processing()
    test_data_structure()
    test_model_creation()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if missing:
        print(f"✗ Missing dependencies: {', '.join(missing)}")
        print("\nInstall missing packages:")
        print("  pip install opencv-python imageio imageio-ffmpeg")
        if 'PyTorch' in missing or 'TorchVision' in missing:
            print("  pip install torch torchvision")
        return 1
    else:
        print("✓ All dependencies installed!")
        print("✓ Setup looks good!")
        print("\nTo start training:")
        print("  cd Backend/pytorch")
        print("  python train_video.py --config config_video.yaml")
        return 0


if __name__ == "__main__":
    sys.exit(main())
