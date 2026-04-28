#!/usr/bin/env python3
"""Test the model output directly to see the bias."""

import torch
from utils.model_utils import DeepfakeDetector

# Load the model
device = torch.device('cpu')
model, device = DeepfakeDetector.load_model('../models/xception_deepfake.pth', device)
model.eval()

# Test with real images
print('Testing REAL images:')
for i in range(5):
    tensor, prep_time = DeepfakeDetector.preprocess_image(f'../DATA/Real/Real_{i}.jpg')
    result = DeepfakeDetector.predict_image(model, tensor, device)
    output = result['confidence_raw']
    pred = result['prediction']
    conf = result['confidence']
    print(f'  Real_{i}.jpg: output={output:.4f}, pred={pred}, conf={conf:.2f}%')

print('\nTesting FAKE images:')
for i in range(5):
    tensor, prep_time = DeepfakeDetector.preprocess_image(f'../DATA/Fake/Fake_{i}.jpg')
    result = DeepfakeDetector.predict_image(model, tensor, device)
    output = result['confidence_raw']
    pred = result['prediction']
    conf = result['confidence']
    print(f'  Fake_{i}.jpg: output={output:.4f}, pred={pred}, conf={conf:.2f}%')

print('\n--- Analysis ---')
print('With inverted logic:')
print('- output < 0.5 = Fake (deepfake)')
print('- output > 0.5 = Real (authentic)')

