import cv2
import numpy as np
import os
import requests
from pathlib import Path

# Create a 1-second video (30 frames) with alternating colors
width, height = 640, 480
fps = 30
video_path = "test_validate.mp4"

out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

for i in range(fps):
    # Alternating noise frames to simulate some activity
    frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    out.write(frame)

out.release()
print(f"Generated test video: {video_path}")

# Test the React Backend
url = "http://localhost:3001/api/upload"
print(f"Testing Backend at {url}...")
try:
    with open(video_path, 'rb') as f:
        files = {'file': ('test_validate.mp4', f, 'video/mp4')}
        response = requests.post(url, files=files)
        print("Response:")
        print(response.json())
except Exception as e:
    print(f"Failed to test Backend: {e}")

# Cleanup
if os.path.exists(video_path):
    os.remove(video_path)
