"""
Video-based training script for deepfake detection
Extracts frames from videos and trains the model on temporal sequences
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
try:
    from torchvision.models import efficientnet_b0
except ImportError:
    from torchvision.models.efficientnet import efficientnet_b0
import cv2
import os
import yaml
import argparse
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import random
from pathlib import Path



def replace_activations_with_relu(module: nn.Module) -> None:
    """Helper to replace all SiLU/Swish activations with ReLU"""
    for name, child in module.named_children():
        if isinstance(child, nn.SiLU):
            setattr(module, name, nn.ReLU(inplace=True))
        else:
            replace_activations_with_relu(child)

class VideoDeepfakeDataset(Dataset):
    """Dataset for video-based deepfake detection"""
    def __init__(self, root_dir, split='train', transform=None, frames_per_video=10, split_ratio=0.8):
        self.root_dir = root_dir
        self.transform = transform
        self.frames_per_video = frames_per_video
        self.split_ratio = split_ratio
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
        
        # Shuffle and split (70% train, 20% val, 10% test)
        random.seed(42)
        random.shuffle(all_samples)
        
        n = len(all_samples)
        train_idx = int(0.7 * n)
        val_idx = int(0.9 * n) # 70% + 20%
        
        if split == 'train':
            self.samples = all_samples[:train_idx]
        elif split == 'val':
            self.samples = all_samples[train_idx:val_idx]
        elif split == 'test':
            self.samples = all_samples[val_idx:]
        
        print(f"Loaded {len(self.samples)} videos for {split} split")
        if len(self.samples) > 0:
            real_count = sum(1 for _, l, _ in self.samples if l == 0)
            fake_count = sum(1 for _, l, _ in self.samples if l == 1)
            print(f"Real: {real_count}, Fake: {fake_count}")
    
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
        
        try:
            frames = self.extract_frames(video_path, self.frames_per_video)
            
            if frames is None or len(frames) == 0:
                # Return default frames if extraction fails
                print(f"Warning: Could not extract frames from {video_path}, using placeholder")
                frames = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(self.frames_per_video)]
            
            # Pad or truncate to exact number of frames
            if len(frames) < self.frames_per_video:
                frames.extend([frames[-1]] * (self.frames_per_video - len(frames)))
            else:
                frames = frames[:self.frames_per_video]
            
            # Convert to PIL Images and apply transform
            processed_frames = []
            for frame in frames:
                from PIL import Image
                pil_image = Image.fromarray(frame)
                
                if self.transform:
                    pil_image = self.transform(pil_image)
                
                processed_frames.append(pil_image)
            
            # Stack frames: [frames, channels, height, width]
            frame_stack = torch.stack(processed_frames, dim=0)
            
            return frame_stack, label, os.path.basename(video_path)
        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
            # Return empty stack as fallback
            empty_frames = torch.zeros(self.frames_per_video, 3, 224, 224)
            return empty_frames, label, os.path.basename(video_path)


class TemporalDeepfakeDetector(nn.Module):
    """Temporal model for video deepfake detection using 3D CNN or frame averaging"""
    def __init__(self, num_frames=10, use_3d=False):
        super(TemporalDeepfakeDetector, self).__init__()
        self.num_frames = num_frames
        self.use_3d = use_3d
        
        # Base CNN for frame processing
        # Using EfficientNet-B0 to match the inference engine
        self.backbone = efficientnet_b0(pretrained=True)
        replace_activations_with_relu(self.backbone)
        
        # EfficientNet-B0 features: 1280
        num_features = 1280
        
        # Remove classification head
        self.backbone.classifier = nn.Identity()
        
        # Temporal modeling
        if use_3d:
            # 3D convolution for temporal modeling
            self.temporal_conv = nn.Conv3d(num_features, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.temporal_pool = nn.AdaptiveAvgPool3d(1)
            temporal_features = 512
        else:
            # LSTM for temporal modeling
            self.temporal_lstm = nn.LSTM(input_size=num_features, hidden_size=256, 
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
        features = self.backbone(x)  # [batch_size*num_frames, 1280]
        features = features.view(batch_size * num_frames, -1)
        
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
            _, (hidden, _) = self.temporal_lstm(features)
            features = hidden[-1]  # [batch_size, 256]
        
        # Classification
        output = self.classifier(features)
        return output


def get_transforms(config):
    """Get data transforms for video frames"""
    aug_config = config.get('augmentation', {})
    resize_size = aug_config.get('resize', 224)
    
    transforms_list = [
        transforms.Lambda(lambda x: x.resize((resize_size, resize_size)) if isinstance(x, type(None)) is False else x),
        transforms.RandomHorizontalFlip() if aug_config.get('random_horizontal_flip', False) else transforms.Lambda(lambda x: x),
    ]
    
    final_transforms = transforms_list + [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=aug_config.get('normalize_mean', [0.485, 0.456, 0.406]),
            std=aug_config.get('normalize_std', [0.229, 0.224, 0.225])
        )
    ]
    
    return transforms.Compose(final_transforms)


def train_epoch(model, train_loader, criterion, optimizer, device, config):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    predictions = []
    targets = []
    
    progress_bar = tqdm(train_loader, desc="Training")
    
    for frames, labels, _ in progress_bar:
        frames = frames.to(device)
        labels = labels.to(device).float().unsqueeze(1)
        
        optimizer.zero_grad()
        
        outputs = model(frames)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # For metrics
        predictions.extend((outputs > 0.5).detach().cpu().numpy().flatten())
        targets.extend(labels.detach().cpu().numpy().flatten())
        
        progress_bar.set_postfix({'loss': total_loss / len(train_loader)})
    
    accuracy = accuracy_score(targets, predictions)
    
    return total_loss / len(train_loader), accuracy


def validate(model, val_loader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for frames, labels, _ in tqdm(val_loader, desc="Validating"):
            frames = frames.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            
            outputs = model(frames)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            predictions.extend((outputs > 0.5).detach().cpu().numpy().flatten())
            targets.extend(labels.detach().cpu().numpy().flatten())
    
    accuracy = accuracy_score(targets, predictions)
    precision = precision_score(targets, predictions, zero_division=0)
    recall = recall_score(targets, predictions, zero_division=0)
    f1 = f1_score(targets, predictions, zero_division=0)
    
    return total_loss / len(val_loader), accuracy, precision, recall, f1


def train_video_model(config):
    """Main training loop for video-based model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load configuration
    data_path = config.get('train_data_path', '../../DATA')
    num_frames = config.get('augmentation', {}).get('num_frames_per_video', 10)
    batch_size = config.get('batch_size', 8)
    num_epochs = config.get('num_epochs', 20)
    learning_rate = config.get('learning_rate', 0.0001)
    split_ratio = config.get('split_ratio', 0.8)
    
    # Verify data exists
    if not os.path.exists(data_path):
        print(f"Error: Data path not found: {data_path}")
        return
    
    real_path = os.path.join(data_path, 'Real')
    fake_path = os.path.join(data_path, 'Fake')
    
    if not os.path.exists(real_path):
        print(f"Warning: Real videos directory not found at {real_path}")
        print("Please ensure your data structure is:")
        print("  DATA/")
        print("    Real/    (real video files)")
        print("    Fake/    (deepfake video files)")
        return
    
    if not os.path.exists(fake_path):
        print(f"Warning: Fake videos directory not found at {fake_path}")
        print("Please ensure your data structure is:")
        print("  DATA/")
        print("    Real/    (real video files)")
        print("    Fake/    (deepfake video files)")
        return
    
    print(f"Loading dataset from: {data_path}")
    print(f"Frames per video: {num_frames}")
    print(f"Batch size: {batch_size}\n")
    
    # Create datasets
    transform = get_transforms(config)
    
    train_dataset = VideoDeepfakeDataset(data_path, split='train', transform=transform, 
                                        frames_per_video=num_frames)
    val_dataset = VideoDeepfakeDataset(data_path, split='val', transform=transform, 
                                      frames_per_video=num_frames)
    test_dataset = VideoDeepfakeDataset(data_path, split='test', transform=transform, 
                                       frames_per_video=num_frames)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Create model
    print("Creating temporal deepfake detector model...")
    model = TemporalDeepfakeDetector(num_frames=num_frames, use_3d=False)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Training loop
    print("\nStarting training...")
    print("=" * 60)
    
    best_val_accuracy = 0
    patience = 5
    patience_counter = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, config)
        val_loss, val_acc, val_prec, val_rec, val_f1 = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        print(f"Val Precision: {val_prec:.4f} | Val Recall: {val_rec:.4f} | Val F1: {val_f1:.4f}")
        
        # Save best model
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            patience_counter = 0
            
            model_save_path = config.get('model_save_path', '../../models/video_deepfake.pth')
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save(model.state_dict(), model_save_path)
            print(f"✓ Model saved: {model_save_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered (no improvement for {patience} epochs)")
                break
    
    # Final evaluation on the test set
    print("\n" + "="*60)
    print("FINAL TEST EVALUATION")
    print("="*60)
    
    model_save_path = config.get('model_save_path', '../../models/video_deepfake.pth')
    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path))
        print(f"Loaded best model from {model_save_path} for testing")
        
    test_loss, test_acc, test_prec, test_rec, test_f1 = validate(model, test_loader, criterion, device)
    
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
    print(f"Test Precision: {test_prec:.4f} | Test Recall: {test_rec:.4f} | Test F1: {test_f1:.4f}")
    
    print("\n" + "=" * 60)
    print(f"Training complete! Best validation accuracy: {best_val_accuracy:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train video-based deepfake detector')
    parser.add_argument('--config', type=str, default='config_video.yaml', 
                       help='Path to training config YAML file')
    
    args = parser.parse_args()
    
    # Load config
    if not os.path.exists(args.config):
        print(f"Config file not found: {args.config}")
        print("Creating default config...")
        default_config = {
            'use_huggingface_dataset': False,
            'train_data_path': '../../DATA',
            'val_data_path': '../../DATA',
            'model_save_path': '../../models/video_deepfake.pth',
            'batch_size': 4,
            'learning_rate': 0.0001,
            'num_epochs': 20,
            'weight_decay': 1e-5,
            'augmentation': {
                'num_frames_per_video': 10,
                'random_horizontal_flip': True,
                'resize': 224,
                'normalize_mean': [0.485, 0.456, 0.406],
                'normalize_std': [0.229, 0.224, 0.225]
            }
        }
        with open('config_video.yaml', 'w') as f:
            yaml.dump(default_config, f)
        config = default_config
    else:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    
    train_video_model(config)
