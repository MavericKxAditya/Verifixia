#!/usr/bin/env python3
"""
Script to analyze the Xception deepfake detection model architecture
"""

import torch
import torch.nn as nn
import os
import sys

def analyze_model():
    """Analyze the Xception model architecture"""
    device = torch.device('cpu')

    # Load the model using the same method as app.py
    model_path = '../models/xception_deepfake.pth'
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return

    try:
        # Import the model class
        sys.path.append('.')
        from utils.model_utils import DeepfakeDetector

        model = DeepfakeDetector(use_pretrained=True)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        print("Model loaded successfully")
        print(f"Model type: {type(model)}")

        print("\n=== MODEL ARCHITECTURE ===")
        print(model)

        print("\n=== DETAILED LAYER ANALYSIS ===")

        def print_layer_info(module, name='', indent=0):
            prefix = '  ' * indent
            if isinstance(module, nn.Conv2d):
                print(f"{prefix}{name}: Conv2d({module.in_channels}→{module.out_channels}, kernel={module.kernel_size}, stride={module.stride}, padding={module.padding})")
                if hasattr(module, 'weight') and module.weight is not None:
                    print(f"{prefix}  Parameters: {module.weight.numel()}")
                    print(f"{prefix}  Filters: {module.out_channels}")
            elif isinstance(module, nn.BatchNorm2d):
                print(f"{prefix}{name}: BatchNorm2d({module.num_features})")
            elif isinstance(module, nn.Linear):
                print(f"{prefix}{name}: Linear({module.in_features}→{module.out_features})")
                print(f"{prefix}  Parameters: {module.weight.numel() + (module.bias.numel() if module.bias is not None else 0)}")
            elif isinstance(module, nn.ReLU):
                print(f"{prefix}{name}: ReLU(inplace={module.inplace})")
            elif isinstance(module, nn.Dropout):
                print(f"{prefix}{name}: Dropout(p={module.p})")
            elif isinstance(module, nn.AdaptiveAvgPool2d):
                print(f"{prefix}{name}: AdaptiveAvgPool2d({module.output_size})")
            elif isinstance(module, nn.Sequential):
                print(f"{prefix}{name}: Sequential")
                for i, child in enumerate(module.children()):
                    print_layer_info(child, f"[{i}]", indent + 1)
            elif hasattr(module, '_modules') and len(module._modules) > 0:
                print(f"{prefix}{name}: {type(module).__name__}")
                for child_name, child in module.named_children():
                    print_layer_info(child, child_name, indent + 1)
            else:
                print(f"{prefix}{name}: {type(module).__name__}")

        print_layer_info(model)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print("\n=== PARAMETER COUNT ===")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

    except Exception as e:
        print(f"Error analyzing model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_model()