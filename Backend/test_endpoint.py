"""Test the /api/upload endpoint"""

import requests
import os
from PIL import Image
import time

# Create test image
test_image_path = "test_upload_image.jpg"
img = Image.new('RGB', (299, 299), color=(100, 150, 200))
img.save(test_image_path)

try:
    print("=" * 70)
    print("TESTING /api/upload ENDPOINT")
    print("=" * 70)
    
    # Test with image field
    print("\n1. Testing with 'image' field...")
    with open(test_image_path, 'rb') as f:
        files = {'image': (test_image_path, f, 'image/jpeg')}
        response = requests.post('http://localhost:3001/api/upload', files=files, timeout=30)
    
    print(f"Status Code: {response.status_code}")
    print(f"Response Headers: {dict(response.headers)}")
    print(f"Response Content: {response.text[:500]}")
    
    if response.status_code == 200:
        print("\n✓ Image upload successful!")
        result = response.json()
        print(f"Prediction: {result.get('prediction')}")
        print(f"Confidence: {result.get('confidence')}")
    else:
        print(f"\n✗ Upload failed with status {response.status_code}")
        
    # Test with file field
    print("\n" + "-" * 70)
    print("2. Testing with 'file' field...")
    with open(test_image_path, 'rb') as f:
        files = {'file': (test_image_path, f, 'image/jpeg')}
        response = requests.post('http://localhost:3001/api/upload', files=files, timeout=30)
    
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        print("✓ File upload successful!")
    else:
        print(f"✗ Upload failed with status {response.status_code}: {response.text[:200]}")

finally:
    if os.path.exists(test_image_path):
        os.remove(test_image_path)
        print("\n✓ Cleaned up test image")
