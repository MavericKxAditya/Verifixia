import torch
import torch.nn as nn
from torchvision import models

try:
    from torchvision.models import efficientnet_b0
except ImportError:
    from torchvision.models.efficientnet import efficientnet_b0

def replace_activations_with_relu(module: nn.Module) -> None:
    """Recursively replaces all SiLU/Swish activations with ReLU"""
    for name, child in module.named_children():
        if isinstance(child, nn.SiLU):
            setattr(module, name, nn.ReLU(inplace=True))
        else:
            replace_activations_with_relu(child)

class VerifixiaEfficientNetLSTM(nn.Module):
    """
    Standard Verifixia Deepfake Detection Architecture:
    EfficientNet-B0 Backbone (ReLU) + LSTM Temporal Layer
    """
    def __init__(self, num_classes=1, num_layers=2, hidden_size=256, dropout=0.3):
        super(VerifixiaEfficientNetLSTM, self).__init__()
        
        # 1. Feature Extractor (Backbone)
        self.backbone = efficientnet_b0(pretrained=True)
        self.backbone.classifier = nn.Identity()
        
        # Force ReLU for consistency across all training and inference
        replace_activations_with_relu(self.backbone)
        
        # 2. Temporal Analysis (LSTM)
        # EfficientNet-B0 outputs 1280 features
        self.lstm = nn.LSTM(
            input_size=1280, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 3. Classification Head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        """
        Input: [batch_size, num_frames, channels, height, width]
        """
        batch_size, num_frames, c, h, w = x.shape
        
        # Fold frames into batch dimension for processing
        x = x.view(batch_size * num_frames, c, h, w)
        features = self.backbone(x) # [batch_size * num_frames, 1280]
        
        # Unfold back to temporal sequence
        features = features.view(batch_size, num_frames, -1)
        
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(features)
        
        # Use the hidden state of the last layer
        last_hidden = hidden[-1]
        
        # Output prediction
        return self.classifier(last_hidden)

if __name__ == "__main__":
    # Quick architecture test
    model = VerifixiaEfficientNetLSTM()
    test_input = torch.randn(2, 8, 3, 224, 224)
    output = model(test_input)
    print(f"Architecture Test Success!")
    print(f"Input Shape: {test_input.shape}")
    print(f"Output Shape: {output.shape}")
