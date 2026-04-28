#!/usr/bin/env python3
"""
Test script to evaluate model accuracy on the Verifixia dataset.
Tests the model directly without requiring a backend server.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import os
import json
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class DeepfakeDataset(Dataset):
    """Custom dataset for deepfake detection"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        # Load Real Images (Label 0)
        real_dir = os.path.join(root_dir, 'Real')
        fake_dir = os.path.join(root_dir, 'Fake')
        
        if os.path.exists(real_dir):
            for img_file in sorted(os.listdir(real_dir))[:50]:
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(real_dir, img_file), 0, 'real'))
        
        # Load Deepfake Images (Label 1)
        if os.path.exists(fake_dir):
            for img_file in sorted(os.listdir(fake_dir))[:50]:
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(fake_dir, img_file), 1, 'fake'))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, label_name = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label, os.path.basename(img_path), label_name
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            image = Image.new('RGB', (224, 224), color=(0, 0, 0))
            if self.transform:
                image = self.transform(image)
            return image, label, os.path.basename(img_path), label_name

class DeepfakeDetectorPretrained(nn.Module):
    """Pre-trained ResNet50-based deepfake detection model"""
    def __init__(self):
        super(DeepfakeDetectorPretrained, self).__init__()
        self.backbone = models.resnet50(pretrained=False)
        num_features = self.backbone.fc.in_features
        # Replace the final layer with our custom classifier
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

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    print("Starting accuracy test...")

    # Data transforms
    test_transform = transforms.Compose([
        transforms.Lambda(lambda x: x.resize((224, 224)) if not isinstance(x, type(None)) else x),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Load dataset (50 real + 50 fake)
    data_path = "../DATA"
    test_dataset = DeepfakeDataset(data_path, transform=test_transform)
    
    if len(test_dataset) == 0:
        print("ERROR: No test images found!")
        return

    print(f"Testing {len(test_dataset)} images (50 real + 50 fake)\n")
    
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
    results = []
    
    with torch.no_grad():
        for inputs, labels, filenames, label_names in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            # Apply sigmoid to get probabilities
            probabilities = torch.sigmoid(outputs).squeeze()
            preds = (probabilities > 0.5).int()
            
            # Record results
            for i in range(len(filenames)):
                pred_label = 'fake' if preds[i].item() > 0.5 else 'real'
                confidence = probabilities[i].item() * 100
                is_correct = pred_label == label_names[i]
                
                results.append({
                    'filename': filenames[i],
                    'expected': label_names[i],
                    'prediction': pred_label,
                    'confidence': round(confidence, 2),
                    'correct': is_correct,
                    'success': True
                })
                
                status = '✓ PASS' if is_correct else '✗ FAIL'
                label_type = label_names[i].upper()
                print(f"{label_type}: {filenames[i]:<20} -> {pred_label:<5} (conf: {confidence:>6.2f}%) {status}")
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    correct = sum(1 for r in results if r['correct'])
    
    # Print summary
    print("\n" + "="*60)
    print("ACCURACY TEST SUMMARY")
    print("="*60)
    print(f"Total Tests:       {len(results)}")
    print(f"Correct:           {correct}/{len(results)}")
    print(f"Accuracy:          {accuracy*100:.2f}%")
    print(f"Precision:         {precision*100:.2f}%")
    print(f"Recall:            {recall*100:.2f}%")
    print(f"F1 Score:          {f1:.4f}")
    print("="*60)

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)

    print("\n" + "="*60)
    print("CONFUSION MATRIX")
    print("     Predicted")
    print("           Real    Fake")
    print("-"*40)
    print(f"Actual Real  | {cm[0,0]:4d}  | {cm[0,1]:4d}  |")
    print(f"      Fake  | {cm[1,0]:4d}  | {cm[1,1]:4d}  |")
    print("="*60)

    # Save results
    with open('accuracy_test_results.json', 'w') as f:
        json.dump({
            'total_tests': len(results),
            'correct_predictions': correct,
            'accuracy': round(float(accuracy)*100, 2),
            'precision': round(float(precision)*100, 2),
            'recall': round(float(recall)*100, 2),
            'f1_score': round(float(f1), 4),
            'results': results
        }, f, indent=2)

    print("\nResults saved to accuracy_test_results.json")

if __name__ == "__main__":
    main()