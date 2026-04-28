"""Diagnostic script to test media upload and analysis pipeline"""

import os
import sys
from PIL import Image
import json

# Test 1: Check if uploads folder exists
print("=" * 70)
print("TEST 1: Checking upload folder")
print("=" * 70)
upload_folder = "uploads"
if not os.path.exists(upload_folder):
    os.makedirs(upload_folder)
    print(f"✓ Created {upload_folder} directory")
else:
    print(f"✓ {upload_folder} directory exists")

# Test 2: Create a test image
print("\n" + "=" * 70)
print("TEST 2: Creating test image")
print("=" * 70)
test_image_path = os.path.join(upload_folder, "test_real.jpg")
try:
    img = Image.new('RGB', (299, 299), color=(100, 150, 200))  # Natural-looking blue
    img.save(test_image_path)
    print(f"✓ Created test image: {test_image_path}")
except Exception as e:
    print(f"✗ Failed to create test image: {e}")
    sys.exit(1)

# Test 3: Test predict_deepfake directly
print("\n" + "=" * 70)
print("TEST 3: Testing predict_deepfake function")
print("=" * 70)
try:
    from app import predict_deepfake, PYTORCH_AVAILABLE, SKLEARN_AVAILABLE
    
    print(f"PyTorch Available: {PYTORCH_AVAILABLE}")
    print(f"Sklearn Available: {SKLEARN_AVAILABLE}")
    
    result = predict_deepfake(test_image_path)
    print("✓ Prediction successful!")
    print(json.dumps(result, indent=2, default=str))
    
except Exception as e:
    print(f"✗ Prediction failed: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Test video prediction
print("\n" + "=" * 70)
print("TEST 4: Testing predict_deepfake_video function")
print("=" * 70)
try:
    from app import predict_deepfake_video
    
    # Create a test video-like file (just for testing the function)
    test_video_path = os.path.join(upload_folder, "test_video.mp4")
    
    # Copy the test image as a "video"
    import shutil
    shutil.copy(test_image_path, test_video_path)
    
    try:
        prediction, confidence = predict_deepfake_video(test_video_path)
        print(f"✓ Video prediction successful!")
        print(f"  Prediction: {prediction}")
        print(f"  Confidence: {confidence}")
    except Exception as ve:
        print(f"⚠ Video prediction attempted (may fail on non-video file)")
        print(f"  Error: {ve}")
        
finally:
    if os.path.exists(test_video_path):
        os.remove(test_video_path)

# Test 5: Test heuristic analysis features
print("\n" + "=" * 70)
print("TEST 5: Testing heuristic analysis features")
print("=" * 70)
try:
    from PIL import ImageStat
    import numpy as np
    from scipy import ndimage
    
    img_rgb = Image.open(test_image_path).convert("RGB")
    img_gray = img_rgb.convert("L")
    
    # Test all features
    stat = ImageStat.Stat(img_gray)
    mean = stat.mean[0]
    stddev = stat.stddev[0]
    print(f"✓ Mean: {mean:.2f}, StdDev: {stddev:.2f}")
    
    # Test color analysis
    r_band = np.array(img_rgb.split()[0]).astype(float)
    g_band = np.array(img_rgb.split()[1]).astype(float)
    b_band = np.array(img_rgb.split()[2]).astype(float)
    print(f"✓ Color bands extracted successfully")
    
    # Test edge detection
    edges = ndimage.laplace(np.array(img_gray))
    print(f"✓ Edge detection successful")
    
    # Test block variance
    w, h = img_gray.size
    blocks = []
    for y in range(0, min(h-8, 64), 8):
        for x in range(0, min(w-8, 64), 8):
            block = np.array(img_gray.crop((x, y, x+8, y+8)))
            blocks.append(np.var(block))
    print(f"✓ Block analysis: {len(blocks)} blocks sampled")
    
except Exception as e:
    print(f"✗ Feature analysis failed: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# Cleanup
print("\n" + "=" * 70)
print("CLEANUP")
print("=" * 70)
if os.path.exists(test_image_path):
    os.remove(test_image_path)
    print(f"✓ Cleaned up test image")

print("\n" + "=" * 70)
print("ALL TESTS COMPLETED")
print("=" * 70)
