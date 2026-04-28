import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
try:
    from torchvision.models import efficientnet_b0
except ImportError:
    from torchvision.models.efficientnet import efficientnet_b0
from PIL import Image, ImageEnhance
import os
import yaml
import argparse
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import random

class DeepfakeDataset(Dataset):
    """Custom dataset for deepfake detection with advanced augmentation"""
    def __init__(self, root_dir, split='train', transform=None, use_huggingface=False):
        self.root_dir = root_dir
        self.transform = transform
        self.use_huggingface = use_huggingface
        self.samples = []

        if not use_huggingface:
            # Fixed: Updated to match actual directory structure (Real/ and Fake/)
            real_dir = os.path.join(root_dir, 'Real')
            fake_dir = os.path.join(root_dir, 'Fake')
            
            all_samples = []
            
            # Load Real Images (Label 0)
            if os.path.exists(real_dir):
                for img_file in os.listdir(real_dir):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        all_samples.append((os.path.join(real_dir, img_file), 0))
            else:
                print(f"Warning: Real images directory not found at {real_dir}")

            # Load Deepfake Images (Label 1)
            if os.path.exists(fake_dir):
                for img_file in os.listdir(fake_dir):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        all_samples.append((os.path.join(fake_dir, img_file), 1))
            else:
                print(f"Warning: Deepfake images directory not found at {fake_dir}")

            # Shuffle and Split (70% Train, 20% Val, 10% Test)
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
            
            print(f"Loaded {len(self.samples)} images for {split} split")
            if len(self.samples) > 0:
                real_count = sum(1 for _, l in self.samples if l == 0)
                fake_count = sum(1 for _, l in self.samples if l == 1)
                print(f"Real: {real_count}, Fake: {fake_count}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (299, 299), color=(0, 0, 0))
            if self.transform:
                image = self.transform(image)
            return image, label

def replace_activations_with_relu(module: nn.Module) -> None:
    """Helper to replace all SiLU/Swish activations with ReLU"""
    for name, child in module.named_children():
        if isinstance(child, nn.SiLU):
            setattr(module, name, nn.ReLU(inplace=True))
        else:
            replace_activations_with_relu(child)

class DeepfakeDetectorEfficient(nn.Module):
    """EfficientNet-B0 based deepfake detection model"""
    def __init__(self):
        super(DeepfakeDetectorEfficient, self).__init__()
        self.backbone = efficientnet_b0(pretrained=True)
        self.backbone.classifier = nn.Identity()
        replace_activations_with_relu(self.backbone)
        
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1280, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.fc(features)

def mixup_batch(inputs, labels, alpha=0.2):
    """Apply Mixup augmentation to a batch"""
    batch_size = inputs.size(0)
    index = torch.randperm(batch_size)
    
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    
    mixed_inputs = lam * inputs + (1 - lam) * inputs[index]
    mixed_labels = lam * labels + (1 - lam) * labels[index]
    
    return mixed_inputs, mixed_labels

def get_augmentation_transforms(config, is_train=True):
    """Get data augmentation transforms with compatibility for older torchvision versions"""
    aug_config = config.get('augmentation', {})
    resize_size = aug_config.get('resize', 299)
    
    if is_train:
        transforms_list = [
            transforms.Lambda(lambda x: x.resize((resize_size, resize_size)) if not isinstance(x, type(None)) else x),
            transforms.RandomHorizontalFlip(),
        ]
    else:
        transforms_list = [
            transforms.Lambda(lambda x: x.resize((resize_size, resize_size)) if not isinstance(x, type(None)) else x),
        ]
    
    # Convert to tensor and normalize
    final_transforms = transforms_list + [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=aug_config.get('normalize_mean', [0.485, 0.456, 0.406]),
            std=aug_config.get('normalize_std', [0.229, 0.224, 0.225])
        )
    ]
    
    return transforms.Compose(final_transforms)

def get_class_weights(dataset):
    """Calculate pos_weight for imbalanced binary classification"""
    labels = np.array([label for _, label in dataset.samples])
    num_neg = np.sum(labels == 0)
    num_pos = np.sum(labels == 1)
    if num_pos == 0:
        pos_weight = 1.0
    else:
        pos_weight = num_neg / num_pos
    return torch.tensor(pos_weight, dtype=torch.float32)

def train_model(config):
    """Train the deepfake detection model with improvements"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data transforms
    train_transform = get_augmentation_transforms(config, is_train=True)
    val_transform = get_augmentation_transforms(config, is_train=False)

    # Create datasets
    train_dataset = DeepfakeDataset(config['train_data_path'], split='train', transform=train_transform, use_huggingface=False)
    val_dataset = DeepfakeDataset(config['val_data_path'], split='val', transform=val_transform, use_huggingface=False)
    test_dataset = DeepfakeDataset(config['val_data_path'], split='test', transform=val_transform, use_huggingface=False)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=int(config['batch_size']), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=int(config['batch_size']), shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=int(config['batch_size']), shuffle=False, num_workers=0)

    # Initialize model
    print("Loading EfficientNet-B0 model...")
    model = DeepfakeDetectorEfficient()
    
    model.to(device)
    
    # Calculate pos_weight for imbalanced data
    pos_weight = None
    if config.get('use_weighted_loss', True):
        pos_weight = get_class_weights(train_dataset)
        pos_weight = pos_weight.to(device)
        print(f"Positive class weight: {pos_weight.item():.4f}")

    # Loss function with pos_weight for binary classification
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) if pos_weight is not None else nn.BCEWithLogitsLoss()
    
    # Optimizer with weight decay (L2 regularization)
    lr = float(config.get('learning_rate', 0.001))
    weight_decay = float(config.get('weight_decay', 1e-4))
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler_type = config.get('lr_scheduler', 'cosine')
    num_epochs = int(config.get('num_epochs', 100))
    if scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif scheduler_type == 'exponential':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=int(config['lr_schedule'].get('step_size', 30)), gamma=float(config['lr_schedule'].get('gamma', 0.1)))

    # Early stopping
    patience = int(config.get('early_stopping', {}).get('patience', 15))
    patience_counter = 0

    # Training loop
    best_accuracy = 0.0
    mixup_alpha = float(config.get('augmentation', {}).get('mixup_alpha', 0.2))

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")

        # Training phase
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []

        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs = inputs.to(device)
            original_labels = labels.float().to(device)  # Keep original binary labels
            loss_labels = original_labels.unsqueeze(1).to(device)  # For loss calculation

            # Apply mixup for loss only (for augmentation)
            if mixup_alpha > 0 and random.random() < 0.5:
                inputs, loss_labels = mixup_batch(inputs, loss_labels, alpha=mixup_alpha)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, loss_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            # Use original labels for metrics, not mixed labels
            preds = (torch.sigmoid(outputs.squeeze()) > 0.5).cpu().detach().numpy().flatten()
            labels_np = original_labels.cpu().numpy().flatten()
            train_preds.extend(preds)
            train_labels.extend(labels_np)

        train_accuracy = accuracy_score(train_labels, train_preds)
        train_loss = train_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                inputs = inputs.to(device)
                labels = labels.float().unsqueeze(1).to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                preds = (torch.sigmoid(outputs.squeeze()) > 0.5).cpu().numpy().flatten()
                labels_np = labels.squeeze().cpu().numpy().flatten()
                val_preds.extend(preds)
                val_labels.extend(labels_np)

        val_accuracy = accuracy_score(val_labels, val_preds)
        val_precision = precision_score(val_labels, val_preds, zero_division=0)
        val_recall = recall_score(val_labels, val_preds, zero_division=0)
        val_f1 = f1_score(val_labels, val_preds, zero_division=0)
        val_loss = val_loss / len(val_loader)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")

        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), config['model_save_path'])
            print(f"✓ Best model saved with accuracy: {best_accuracy:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

        scheduler.step()

    # Final evaluation on the test set
    print("\n" + "="*30)
    print("FINAL TEST EVALUATION")
    print("="*30)
    
    # Load best model for testing
    model.load_state_dict(torch.load(config['model_save_path']))
    model.eval()
    
    test_preds = []
    test_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = (torch.sigmoid(outputs.squeeze()) > 0.5).cpu().numpy().flatten()
            test_preds.extend(preds)
            test_labels.extend(labels.numpy().flatten())
            
    test_accuracy = accuracy_score(test_labels, test_preds)
    test_precision = precision_score(test_labels, test_preds, zero_division=0)
    test_recall = recall_score(test_labels, test_preds, zero_division=0)
    test_f1 = f1_score(test_labels, test_preds, zero_division=0)
    
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")

    print(f"\nTraining completed. Best validation accuracy: {best_accuracy:.4f}")

def main():
    parser = argparse.ArgumentParser(description='Train Deepfake Detection Model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    args = parser.parse_args()

    # Sanitize and validate the config file path to prevent path traversal
    config_path = os.path.realpath(args.config)
    allowed_base = os.path.realpath(os.path.dirname(os.path.abspath(__file__)))
    if not config_path.startswith(allowed_base):
        raise ValueError(f"Config path '{config_path}' is outside the allowed directory.")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(config['model_save_path']), exist_ok=True)

    # Train the model
    train_model(config)

if __name__ == '__main__':
    main()
