#!/usr/bin/env python3
"""
Direct model accuracy test - no Flask backend required
Tests the trained model directly on the dataset
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from pathlib import Path

class DeepfakeDataset(Dataset):
    """Custom dataset for deepfake detection"""
    def __init__(self, root_dir, split='test', transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        # Load Real Images (Label 0)
        real_dir = os.path.join(root_dir, 'Real')
        fake_dir = os.path.join(root_dir, 'Fake')
        
        if os.path.exists(real_dir):
            for img_file in sorted(os.listdir(real_dir)):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(real_dir, img_file), 0))
        
        # Load Deepfake Images (Label 1)
        if os.path.exists(fake_dir):
            for img_file in sorted(os.listdir(fake_dir)):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(fake_dir, img_file), 1))
        
        print(f"Loaded {len(self.samples)} test images")
        print(f"Real: {sum(1 for _, l in self.samples if l == 0)}, Fake: {sum(1 for _, l in self.samples if l == 1)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label, os.path.basename(img_path)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            image = Image.new('RGB', (224, 224), color=(0, 0, 0))
            if self.transform:
                image = self.transform(image)
            return image, label, os.path.basename(img_path)

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

class DeepfakeDetector(nn.Module):
    """Xception-based deepfake detection model - matches trained weights"""
    def __init__(self):
        super(DeepfakeDetector, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 0)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        # Entry flow
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(64)

        # Middle flow (simplified)
        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(256)

        # Exit flow
        self.conv5 = nn.Conv2d(256, 512, 3, 2, 1)
        self.bn5 = nn.BatchNorm2d(512)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, 1)

    def forward(self, x):
        # Entry flow
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))

        # Middle flow
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))

        # Exit flow
        x = self.relu(self.bn5(self.conv5(x)))

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

def test_model():
    """Test the trained model directly"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Data transforms (resize to 224 for standard CNN compatibility)
    test_transform = transforms.Compose([
        transforms.Lambda(lambda x: x.resize((224, 224)) if not isinstance(x, type(None)) else x),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Load dataset
    data_path = "../DATA"
    test_dataset = DeepfakeDataset(data_path, split='test', transform=test_transform)
    
    if len(test_dataset) == 0:
        print("ERROR: No test images found!")
        return

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    # Load model
    model = DeepfakeDetectorPretrained()
    model_path = "../models/xception_deepfake.pth"
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        return

    print(f"Loading model from {model_path}...")
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("✓ Model loaded successfully\n")
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return

    model.to(device)
    model.eval()

    # Test the model
    all_preds = []
    all_labels = []
    all_confidences = []
    
    print("Running inference on test set...\n")
    
    with torch.no_grad():
        for batch_idx, (inputs, labels, filenames) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            # Apply sigmoid to get probabilities
            probabilities = torch.sigmoid(outputs).squeeze()
            preds = (probabilities > 0.5).int()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(probabilities.cpu().numpy())
            
            if (batch_idx + 1) % 5 == 0:
                print(f"Processed {batch_idx + 1}/{len(test_loader)} batches")

    # Calculate metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    
    # Print results
    print("\n" + "="*60)
    print("ACCURACY TEST RESULTS")
    print("="*60)
    print(f"Total Images Tested: {len(all_labels)}")
    print(f"Correctly Classified: {np.sum(all_preds == all_labels)}/{len(all_labels)}")
    print()
    print(f"Accuracy:  {accuracy*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall:    {recall*100:.2f}%")
    print(f"F1 Score:  {f1:.4f}")
    print()
    print("Confusion Matrix:")
    print(f"  True Negatives (Real Correct):  {tn}")
    print(f"  False Positives (Real as Fake): {fp}")
    print(f"  False Negatives (Fake as Real): {fn}")
    print(f"  True Positives (Fake Correct):  {tp}")
    print()
    print(f"Real Detection Rate (Specificity): {tn/(tn+fp)*100 if (tn+fp) > 0 else 0:.2f}%")
    print(f"Fake Detection Rate (Sensitivity): {tp/(tp+fn)*100 if (tp+fn) > 0 else 0:.2f}%")
    print("="*60)

if __name__ == '__main__':
    test_model()
