# Dataset Collection Quick Reference

## 🚀 Quick Start (Choose ONE method)

### Method 1: Create Sample Videos (FASTEST - 2 minutes)
Perfect for testing if everything works. Creates 2 dummy test videos.

```bash
cd Backend/pytorch
python dataset_setup.py
# Select: "Option 1: Create Sample Videos"
```

**Result:** 2 test videos created in `DATA/Real/` and `DATA/Fake/`

---

### Method 2: Download from YouTube (EASY - 15-30 minutes)
Download real face videos from YouTube.

```bash
# First, install required tool:
pip install yt-dlp

# Then download videos:
python ../scripts/download_youtube_videos.py --url "https://www.youtube.com/watch?v=VIDEO_ID" --category real
```

**Good sources:**
- TED Talks (diverse speakers)
- Interviews
- Vlogs/Podcasts
- News segments
- Educational videos

---

### Method 3: Manual Collection (FLEXIBLE)
Add your own video files:

1. Create folders:
   ```
   DATA/
   ├── Real/
   │   ├── video1.mp4
   │   ├── video2.mov
   │   └── ...
   └── Fake/
       ├── fake1.mp4
       ├── fake2.mp4
       └── ...
   ```

2. Add your videos and train:
   ```bash
   cd Backend/pytorch
   python train_video.py --config config_video.yaml
   ```

---

## 📊 Dataset Requirements

| Dataset Size | Training Time | Model Quality | Use Case |
|---------------|---------------|---------------|----------|
| 2-5 videos each | < 5 min | Testing only | Validation |
| 10-20 videos each | 5-30 min | Good | Development |
| 50-100 videos each | 30-120 min | Excellent | Production |
| 500+ videos each | Hours | Expert-level | Research |

---

## 🎯 Recommended Setup by Use Case

### For Quick Testing
```bash
# 1. Create sample videos
python dataset_setup.py
# Select: Create Sample Videos

# 2. Train immediately
python train_video.py --config config_video.yaml
# Takes ~5 minutes
```

### For Development
```bash
# 1. Download 20-30 videos from YouTube
python ../scripts/download_youtube_videos.py --url "..." --category real
# Repeat for multiple videos

# 2. Add deepfakes from public sources or create your own

# 3. Train model
python train_video.py --config config_video.yaml
# Takes 30-120 minutes depending on size
```

### For Production
```bash
# Download large public datasets:
# - FaceForensics++ (requires registration)
# - DFDC (download from official site)
# - Celeb-DF (download from official site)

# Organize into DATA/Real/ and DATA/Fake/

# Train with optimized config
python train_video.py --config config_video.yaml
```

---

## 📥 All Available Tools

### 1. Interactive Setup Wizard
```bash
cd Backend/pytorch
python dataset_setup.py
```
User-friendly menu for all dataset operations.

### 2. Direct Dataset Downloader
```bash
# Create sample videos
python ../scripts/download_deepfake_dataset.py --create-samples

# Download from online sources
python ../scripts/download_deepfake_dataset.py --download

# List current videos
python ../scripts/download_deepfake_dataset.py --list

# Show collection guide
python ../scripts/download_deepfake_dataset.py --info
```

### 3. YouTube Video Downloader
```bash
# Install tool
pip install yt-dlp

# Download single video
python ../scripts/download_youtube_videos.py --url "https://..." --category real

# Batch download from list
python ../scripts/download_youtube_videos.py --file urls.txt --category real
```

### 4. Training Script
```bash
# Train on collected videos
python train_video.py --config config_video.yaml
```

---

## 🔗 Public Datasets to Download

| Dataset | URL | Size | Real/Fake |
|---------|-----|------|-----------|
| FaceForensics++ | http://kaldir.vc.in.tum.de/faceforensics/ | Large | Both |
| DFDC | https://www.deepfakedetectionchallenge.org/ | Large | Both |
| Celeb-DF | https://www.cs.albany.edu/~lsw/celeb-df.html | Medium | Both |
| DeeperForensics | https://github.com/endymecy/DeeperForensics-1.0 | Large | Both |

---

## ✅ Workflow Checklist

**Step 1: Choose Dataset Method**
- [ ] Quick Sample (2 min)
- [ ] YouTube Download (30 min)
- [ ] Manual Collection (ongoing)
- [ ] Public Dataset (1-2 hours)

**Step 2: Collect/Download Videos**
```bash
# Option A: Create samples
python ../scripts/download_deepfake_dataset.py --create-samples

# Option B: Download from YouTube
pip install yt-dlp
python ../scripts/download_youtube_videos.py --url "..." --category real

# Option C: Manual - copy files to DATA/Real/ and DATA/Fake/
```

**Step 3: Verify Dataset**
```bash
python ../scripts/download_deepfake_dataset.py --list
# Should show: Real: X videos, Fake: X videos
```

**Step 4: Train Model**
```bash
python train_video.py --config config_video.yaml
```

**Step 5: Monitor Training**
- Watch terminal output
- Check validation accuracy increases
- Model auto-saves best version
- Training completes when accuracy plateaus

---

## 🛠️ Troubleshooting

### "No module named 'yt_dlp'"
```bash
pip install yt-dlp
```

### "No videos found in DATA/"
```bash
# Create samples first:
python ../scripts/download_deepfake_dataset.py --create-samples

# Or add your own videos to:
# - DATA/Real/ (real face videos)
# - DATA/Fake/ (deepfake videos)
```

### "Training is very slow"
1. Reduce frames: `num_frames_per_video: 5` in config_video.yaml
2. Reduce batch size: `batch_size: 2`
3. Use GPU if available
4. Start with fewer videos (10-20 first)

### "Out of memory"
```yaml
# In config_video.yaml, reduce:
batch_size: 2          # from 4
num_frames_per_video: 5  # from 10
```

### "Can't download from YouTube"
- Check internet connection
- Try different video URL
- Some videos may be restricted
- Ensure yt-dlp is installed: `pip install --upgrade yt-dlp`

---

## 📝 Tips

✓ **Start small**: Begin with 5-10 videos to test
✓ **Balance**: Keep real/fake videos roughly equal
✓ **Diversity**: Include different people, angles, lighting
✓ **Quality**: Higher resolution = better results (720p+)
✓ **Duration**: 2-30 second videos work best
✓ **Ethics**: Only use videos you have permission to use

---

## Next Steps

1. Choose a dataset method above
2. Run appropriate download/collection command
3. Verify videos are in `DATA/Real/` and `DATA/Fake/`
4. Start training: `python train_video.py`
5. Monitor progress in terminal
6. Use trained model in your app

Questions? Check `VIDEO_TRAINING_GUIDE.md` for detailed information.
