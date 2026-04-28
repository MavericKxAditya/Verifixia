import requests
import os

# Define the API endpoint
url = "http://localhost:3001/api/upload"

# Use the sample image from Hugging Face documentation if available, otherwise any image
# Since we don't have internet access to download, we'll look for a local image in DATA directory
# or generate a dummy image if possible? 
# Wait, user prompt had a URL, but we need a file for upload.
# Let's try to assume we can find an image in DATA or create a dummy one.

from PIL import Image
import io

def create_dummy_image():
    # Create a simple RGB image
    img = Image.new('RGB', (100, 100), color = 'red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)
    return img_byte_arr

def test_upload():
    print("Testing /api/upload endpoint...")
    try:
        # Create a dummy image
        img_data = create_dummy_image()
        files = {'image': ('test_image.jpg', img_data, 'image/jpeg')}
        
        response = requests.post(url, files=files)
        
        if response.status_code == 200:
            print("Success! Response:")
            print(response.json())
        else:
            print(f"Failed with status code {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"Error connecting to server: {e}")

if __name__ == "__main__":
    test_upload()
