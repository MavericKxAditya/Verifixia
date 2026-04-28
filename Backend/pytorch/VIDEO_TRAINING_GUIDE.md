# Video-Based Deepfake Detection Training Guide

## Overview
This guide explains how to train the deepfake detection model specifically on videos (not images). The model will:
- Extract key frames from each video at regular intervals
- Process frames temporally using LSTM-based feature aggregation
- Train to detect deepfakes in video content

## Data Structure

Your data should be organized as follows:

```
DATA/
├── Real/           # Real video files
│   ├── video1.mp4
│   ├── video2.mp4
│   ├── video3.mov
│   └── ...
└── Fake/           # Deepfake video files
    ├── deepfake1.mp4
    ├── deepfake2.mp4
    ├── deepfake3.mov
    └── ...
```

**Supported Video Formats:**
- MP4 (.mp4)
- MOV (.mov)
- AVI (.avi)
- MKV (.mkv)
- WebM (.webm)
- FLV (.flv)
- WMV (.wmv)

## Installation

### 1. Install Video Processing Dependencies

```bash
# Install OpenCV and video handling
pip install opencv-python>=4.8.0
pip install imageio>=2.33.0
pip install imageio-ffmpeg>=0.4.9

# Or install all at once:
cd Backend
pip install -r requirements.txt
```

### 2. Install PyTorch (if not already installed)

**CPU version:**
```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
```

**GPU (CUDA 12.x):**
```bash
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision
```

**GPU (CUDA 11.8):**
```bash
pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision
```

**Apple Silicon (M1/M2/M3):**
```bash
pip install torch torchvision
```

## Training

### Quick Start

```bash
cd Backend/pytorch

# Train with default configuration
python train_video.py --config config_video.yaml
```

### Configuration

Edit `config_video.yaml` to customize:

```yaml
batch_size: 4                           # Adjust based on GPU memory
learning_rate: 0.0001
num_epochs: 20
augmentation:
  num_frames_per_video: 10             # Extract 10 frames per video
  resize: 224                          # Resize frames to 224x224
```

**Key Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 4 | Smaller than image training due to frame stacking |
| `num_frames_per_video` | 10 | Frames extracted per video |
| `learning_rate` | 0.0001 | Learning rate for Adam optimizer |
| `num_epochs` | 20 | Maximum training epochs |
| `resize` | 224 | Frame resolution (224x224 for ResNet50) |

### GPU Memory Requirements

- **4 frames/video, batch_size=4:** ~6 GB VRAM
- **10 frames/video, batch_size=4:** ~8 GB VRAM (recommended)
- **10 frames/video, batch_size=8:** ~16 GB VRAM

**If out of memory:**
1. Reduce `batch_size` (e.g., 4 → 2)
2. Reduce `num_frames_per_video` (e.g., 10 → 5)
3. Reduce frame `resize` (e.g., 224 → 160)

## Training Process

The training script will:

1. **Scan videos:** Find all videos in `DATA/Real/` and `DATA/Fake/`
2. **Split data:** 80% training, 20% validation
3. **Extract frames:** Sample evenly-spaced frames from each video
4. **Train model:** Process frame sequences through temporal model
5. **Save best model:** Automatically save to `models/video_deepfake.pth`

### Training Output

```
Loading dataset from: ../../DATA
Frames per video: 10
Batch size: 4

Loaded 25 videos for train split
Real: 15, Fake: 10
Loaded 6 videos for val split
Real: 3, Fake: 3

Creating temporal deepfake detector model...

Starting training...
============================================================

Epoch 1/20
Training: 100%|████████| 7/7 [00:45<00:00,  6.43s/it]
Train Loss: 0.6932 | Train Acc: 0.5200
Val Loss: 0.6891 | Val Acc: 0.5833
Val Precision: 0.6667 | Val Recall: 0.5000 | Val F1: 0.5714
✓ Model saved: ../../models/video_deepfake.pth

...

Training complete! Best validation accuracy: 0.8333
```

## Tips for Best Results

### 1. **Video Quality**
- Use high-quality videos (480p minimum, 720p or higher recommended)
- Ensure consistent lighting and angles within each video
- Videos should be at least 2-3 seconds long

### 2. **Dataset Balance**
- Keep real and fake videos balanced (roughly equal numbers)
- At least 20 videos of each type for meaningful training
- Ideally 50+ videos of each type for production models

### 3. **Video Diversity**
- Include videos from different sources
- Vary facial poses, angles, lighting conditions
- Include different deepfake methods (face-swap, reenactment, etc.)

### 4. **Training Stability**
- Start with `num_frames_per_video: 5` for quick testing
- Increase to 10-15 frames for production models
- Use separate validation set (automatically handled by script)

### 5. **Monitoring**
- Watch the loss decrease over epochs
- If validation accuracy plateaus, training has likely converged
- Early stopping triggers after 5 epochs without improvement

## Common Issues & Solutions

### Issue: "No such file or directory: DATA/Real"
**Solution:** Ensure data is in correct location:
```bash
# From Backend/pytorch directory, data should be at:
../../DATA/Real/
../../DATA/Fake/
```

### Issue: Out of memory (CUDA)
**Solution:** Reduce batch size or frames:
```yaml
batch_size: 2
num_frames_per_video: 5
```

### Issue: OpenCV can't read video
**Solution:** Install FFmpeg:
```bash
# Windows (via conda or chocolatey)
conda install -c conda-forge ffmpeg

# macOS
brew install ffmpeg

# Linux
sudo apt-get install ffmpeg
```

### Issue: Training very slow
**Solution:** 
- Check GPU usage: `nvidia-smi` (should show 80%+ GPU usage)
- Reduce number of frames: `num_frames_per_video: 5`
- Increase batch size if GPU memory allows

## Using the Trained Model

Once trained, the model is saved to `models/video_deepfake.pth`.

To use in your app, update `Backend/utils/model_utils.py` to load the video model and process video input accordingly.

## Advanced Configuration

### Using 3D CNN Instead of LSTM

```yaml
augmentation:
  use_3d: true  # Uses 3D convolution for temporal modeling
```

3D CNN is faster but requires more VRAM. LSTM is more flexible and recommended for most cases.

### Custom Learning Rate Schedule

```yaml
lr_scheduler: "cosine"  # or "step"
lr_schedule:
  step_size: 20
  gamma: 0.1
```

## Next Steps

1. **Prepare your video data** in `DATA/Real/` and `DATA/Fake/`
2. **Install dependencies**: `pip install -q -r requirements.txt`
3. **Start training**: `python train_video.py`
4. **Monitor progress** and adjust hyperparameters as needed
5. **Integrate trained model** into the app for video detection

## Questions or Issues?

Check `TROUBLESHOOTING.md` for common solutions, or refer to the inline code comments in `train_video.py`.
