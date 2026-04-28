# Model utilities and helper functions

import os
import time
from typing import Dict, Tuple, Optional, Any

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np

_TORCH_AVAILABLE = True


if _TORCH_AVAILABLE:
    class DeepfakeDetector(nn.Module):
        """ResNet50-based deepfake detection model (supports both custom and pre-trained)"""
        def __init__(self, use_pretrained=True):
            super(DeepfakeDetector, self).__init__()
            self.use_pretrained = use_pretrained
            
            if use_pretrained:
                # Load pre-trained ResNet50 (stable across torchvision versions)
                self.backbone = models.resnet50(pretrained=True)
                num_features = self.backbone.fc.in_features
                original_fc = self.backbone.fc
                self.backbone.fc = nn.Sequential(  # type: ignore[assignment]
                    nn.Dropout(0.5),
                    nn.Linear(num_features, 512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.3),
                    nn.Linear(512, 1),
                    nn.Sigmoid()
                )
                del original_fc
            else:
                # Custom simple architecture
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
                self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            if self.use_pretrained:
                return self.backbone(x)
            else:
                # Custom forward pass
                x = self.relu(self.bn1(self.conv1(x)))
                x = self.relu(self.bn2(self.conv2(x)))
                x = self.relu(self.bn3(self.conv3(x)))
                x = self.relu(self.bn4(self.conv4(x)))
                x = self.relu(self.bn5(self.conv5(x)))
                x = self.global_pool(x)
                x = x.view(x.size(0), -1)
                x = self.dropout(x)
                x = self.fc(x)
                return self.sigmoid(x)

        @staticmethod
        def load_model(model_path: str, device: Optional[Any] = None, use_pretrained: bool = True) -> Tuple[Any, Any]:
            """Load a trained model with error handling"""
            if device is None:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            model = DeepfakeDetector(use_pretrained=use_pretrained)
            
            try:
                state_dict = torch.load(model_path, map_location=device)
                model.load_state_dict(state_dict)
                model.to(device)
                model.eval()
                print(f"Model loaded successfully from {model_path}")
            except Exception as e:
                print(f"Error loading model: {e}")
                raise

            return model, device

        @staticmethod
        def preprocess_image(image_path: str, image_size: int = 299) -> Tuple[Any, float]:
            """Preprocess image for model input and return preprocessing time"""
            start_time = time.time()
            
            # Use v2 transforms or manual resizing
            try:
                from torchvision.transforms import v2
                transform = v2.Compose([
                    v2.Resize((image_size, image_size)),
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                image = Image.open(image_path).convert('RGB')
                tensor = transform(image).unsqueeze(0)
            except (ImportError, AttributeError):
                # Fallback to manual preprocessing
                image = Image.open(image_path).convert('RGB')
                image = image.resize((image_size, image_size), Image.BILINEAR)
                
                tensor = transforms.ToTensor()(image)
                tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(tensor)
                tensor = tensor.unsqueeze(0)
            
            preprocessing_time = time.time() - start_time
            return tensor, preprocessing_time

        @staticmethod
        def predict_image(model: Any, image_tensor: Any, device: Any) -> Dict[str, Any]:
            """Make prediction with detailed information"""
            start_time = time.time()
            
            image_tensor = image_tensor.to(device)

            with torch.no_grad():
                outputs = model(image_tensor)
                confidence_raw = outputs.item()
            
            inference_time = time.time() - start_time
            
            # Try inverted logic: maybe the model was trained with labels swapped
            # If confidence_raw > 0.5 means Real instead of Fake
            confidence_raw = outputs.item()
            
            inference_time = time.time() - start_time
            
            # By convention during training we used label=1 for fake
            # images and label=0 for real ones. The sigmoid output therefore
            # represents P(fake).  Previously the code assumed the opposite
            # (inverted logic), which caused real images to be flagged fake.
            #
            # To allow experimentation we also expose a small flag that can be
            # toggled via `MODEL_OUTPUT_IS_REAL` env variable. When true the
            # original inverted behaviour is preserved for legacy models.

            invert = os.getenv("MODEL_OUTPUT_IS_REAL", "false").lower() in ("1","true","yes")
            if invert:
                # legacy behaviour: >0.5 == real
                is_fake = False if confidence_raw > 0.5 else True
                confidence_for_prediction = min(95.0, abs(confidence_raw - 0.5) * 200)
            else:
                # normal behaviour: >0.5 == fake
                is_fake = True if confidence_raw > 0.5 else False
                confidence_for_prediction = min(95.0, abs(confidence_raw - 0.5) * 200)

            # Ensure minimum confidence
            confidence_for_prediction = max(5.0, confidence_for_prediction)
            
            result = {
                "prediction": "Fake" if is_fake else "Real",
                "confidence": round(confidence_for_prediction, 2),
                "confidence_raw": confidence_raw,
                "threat_level": "high" if confidence_for_prediction > 70 else "medium" if confidence_for_prediction > 40 else "low",
                "inference_time_ms": round(inference_time * 1000, 2)
            }
            return result

        @staticmethod
        def get_model_info(model_path: str) -> Dict[str, Any]:
            """Get comprehensive information about the model"""
            info = {
                "model_name": "Verifixia AI Xception",
                "version": "2.4.1",
                "architecture": "Xception-based CNN",
                "input_size": "299x299",
                "framework": "PyTorch",
            }
            
            if os.path.exists(model_path):
                file_size = os.path.getsize(model_path) / (1024 * 1024)  # Size in MB
                info.update({
                    "exists": True,
                    "size_mb": round(file_size, 2),
                    "path": model_path,
                    "status": "loaded"
                })
            else:
                info.update({
                    "exists": False,
                    "path": model_path,
                    "status": "not_found"
                })
            
            return info

        @staticmethod
        def get_model_metadata(model: Any, device: Any) -> Dict[str, Any]:
            """Get detailed model metadata including parameter count"""
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            return {
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "device": str(device),
                "layers": {
                    "convolutional": 5,
                    "batch_norm": 5,
                    "fully_connected": 1,
                    "dropout": 1
                },
                "architecture_details": {
                    "entry_flow": "Conv2d(3→32→64)",
                    "middle_flow": "Conv2d(64→128→256)",
                    "exit_flow": "Conv2d(256→512)",
                    "classifier": "FC(512→1) + Sigmoid"
                }
            }

        @staticmethod
        def interpret_confidence(confidence_raw: float) -> Dict[str, str]:
            """Interpret confidence score and provide detailed analysis"""
            # kept here for backwards compatibility in case someone references
            # ModelUtils.interpret_confidence directly; this simply forwards to
            # the module-level helper defined below.
            return interpret_confidence(confidence_raw)
else:
    class DeepfakeDetector:  # type: ignore[no-redef]
        """Stub when PyTorch is unavailable"""
        pass


# Alias for backward compatibility
ModelUtils = DeepfakeDetector


# ── module helpers ───────────────────────────────────────────────────────────

def interpret_confidence(confidence_raw: float) -> Dict[str, str]:
    """Interpret a raw confidence score and return descriptive metadata.

    This is provided at module level so callers can import it directly and
    static analyzers like Pylance can resolve the symbol without needing to
    look through the conditional class definition above.
    """
    if confidence_raw > 0.9:
        return {
            "level": "Very High",
            "description": "Strong indicators of deepfake manipulation detected",
            "recommendation": "Content should be flagged and reviewed"
        }
    elif confidence_raw > 0.7:
        return {
            "level": "High",
            "description": "Multiple deepfake artifacts identified",
            "recommendation": "Content likely manipulated, further analysis recommended"
        }
    elif confidence_raw > 0.5:
        return {
            "level": "Moderate",
            "description": "Some suspicious patterns detected",
            "recommendation": "Content may be manipulated, manual review suggested"
        }
    elif confidence_raw > 0.3:
        return {
            "level": "Low",
            "description": "Minimal deepfake indicators found",
            "recommendation": "Content appears mostly authentic"
        }
    else:
        return {
            "level": "Very Low",
            "description": "No significant manipulation detected",
            "recommendation": "Content appears authentic"
        }



