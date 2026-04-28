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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import random
from pathlib import Path
from PIL import Image
from architecture import VerifixiaEfficientNetLSTM

# ──────────────────────────────────────────────────────────────────────
# UTILITIES
# ──────────────────────────────────────────────────────────────────────

def mixup_batch(inputs, labels, alpha=0.2):
    """Apply Mixup augmentation to a batch"""
    batch_size = inputs.size(0)
    index = torch.randperm(batch_size)
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    mixed_inputs = lam * inputs + (1 - lam) * inputs[index]
    mixed_labels = lam * labels + (1 - lam) * labels[index]
    return mixed_inputs, mixed_labels

# ──────────────────────────────────────────────────────────────────────
# DATASET
# ──────────────────────────────────────────────────────────────────────

class UnifiedDeepfakeDataset(Dataset):
    """Dataset for both video and image-based deepfake detection"""
    def __init__(self, root_dir, split='train', transform=None, frames_per_video=10):
        self.root_dir = root_dir
        self.transform = transform
        self.frames_per_video = frames_per_video
        self.samples = []
        
        VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv', '.wmv'}
        IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        self.ALL_EXTENSIONS = VIDEO_EXTENSIONS | IMAGE_EXTENSIONS
        
        real_dir = os.path.join(root_dir, 'Real')
        fake_dir = os.path.join(root_dir, 'Fake')
        
        all_samples = []
        
        # Load Files
        for folder, label, label_name in [(real_dir, 0, 'real'), (fake_dir, 1, 'fake')]:
            if os.path.exists(folder):
                for file in os.listdir(folder):
                    ext = Path(file).suffix.lower()
                    if ext in self.ALL_EXTENSIONS:
                        is_image = ext in IMAGE_EXTENSIONS
                        all_samples.append({
                            'path': os.path.join(folder, file),
                            'label': label,
                            'type': 'image' if is_image else 'video'
                        })
            else:
                print(f"Warning: Directory not found at {folder}")
        
        # Shuffle and split (70% train, 20% val, 10% test)
        random.seed(42)
        random.shuffle(all_samples)
        
        n = len(all_samples)
        train_idx = int(0.7 * n)
        val_idx = int(0.9 * n)
        
        if split == 'train':
            self.samples = all_samples[:train_idx]
        elif split == 'val':
            self.samples = all_samples[train_idx:val_idx]
        elif split == 'test':
            self.samples = all_samples[val_idx:]
            
        print(f"Loaded {len(self.samples)} samples for {split} split")

    def __len__(self):
        return len(self.samples)
    
    def extract_frames(self, video_path, num_frames=10):
        frames = []
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < num_frames: num_frames = max(1, total_frames)
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames

    def __getitem__(self, idx):
        item = self.samples[idx]
        try:
            if item['type'] == 'image':
                img = Image.open(item['path']).convert('RGB')
                frames = [np.array(img)] * self.frames_per_video
            else:
                frames = self.extract_frames(item['path'], self.frames_per_video)
            
            if not frames: frames = [np.zeros((224, 224, 3), dtype=np.uint8)] * self.frames_per_video
            while len(frames) < self.frames_per_video: frames.append(frames[-1])
            frames = frames[:self.frames_per_video]
            
            processed = [self.transform(Image.fromarray(f)) if self.transform else torch.from_numpy(f) for f in frames]
            return torch.stack(processed, dim=0), item['label'], os.path.basename(item['path'])
        except Exception as e:
            print(f"Error processing {item['path']}: {e}")
            return torch.zeros(self.frames_per_video, 3, 224, 224), item['label'], ""

# ──────────────────────────────────────────────────────────────────────
# TRAINING
# ──────────────────────────────────────────────────────────────────────

def run_evaluation(model, loader, criterion, device, desc="Evaluating"):
    model.eval()
    total_loss, preds, targets = 0, [], []
    with torch.no_grad():
        for x, y, _ in tqdm(loader, desc=desc):
            x, y = x.to(device), y.to(device).float().unsqueeze(1)
            out = model(x)
            total_loss += criterion(out, y).item()
            preds.extend((out > 0.5).cpu().numpy().flatten())
            targets.extend(y.cpu().numpy().flatten())
    
    acc = accuracy_score(targets, preds)
    return total_loss / len(loader), acc, precision_score(targets, preds, zero_division=0), recall_score(targets, preds, zero_division=0), f1_score(targets, preds, zero_division=0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config_video.yaml')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f: config = yaml.safe_load(f)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    num_frames = config.get('augmentation', {}).get('num_frames_per_video', 8)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    data_path = config.get('train_data_path', '../../DATA')
    train_ds = UnifiedDeepfakeDataset(data_path, 'train', transform, num_frames)
    val_ds = UnifiedDeepfakeDataset(data_path, 'val', transform, num_frames)
    test_ds = UnifiedDeepfakeDataset(data_path, 'test', transform, num_frames)
    
    bs = config.get('batch_size', 8)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=bs)
    test_loader = DataLoader(test_ds, batch_size=bs)
    
    model = VerifixiaEfficientNetLSTM().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.get('learning_rate', 1e-4))
    
    best_acc = 0
    for epoch in range(config.get('num_epochs', 20)):
        model.train()
        train_loss = 0
        for x, y, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            x, y = x.to(device), y.to(device).float().unsqueeze(1)
            
            if random.random() < 0.3: x, y = mixup_batch(x, y)
                
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        v_loss, v_acc, v_prec, v_rec, v_f1 = run_evaluation(model, val_loader, criterion, device, "Validation")
        print(f"Epoch {epoch+1} | Val Acc: {v_acc:.4f} | Val F1: {v_f1:.4f}")
        
        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(model.state_dict(), config.get('model_save_path', 'unified_model.pth'))
            print("✓ Saved Best Model")
            
    # Final Test
    model.load_state_dict(torch.load(config.get('model_save_path', 'unified_model.pth')))
    t_loss, t_acc, t_prec, t_rec, t_f1 = run_evaluation(model, test_loader, criterion, device, "Testing")
    print(f"\nFINAL TEST RESULTS | Acc: {t_acc:.4f} | F1: {t_f1:.4f}")

if __name__ == "__main__":
    main()
