#!/usr/bin/env python3
"""
Interactive Dataset Setup Wizard
Guides you through collecting and organizing a deepfake detection dataset
"""

import os
import sys
from pathlib import Path
import subprocess


def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_option(num, text):
    """Print formatted option"""
    print(f"\n  {num}. {text}")


def get_choice(options):
    """Get user choice"""
    while True:
        try:
            choice = input("\n  → Enter your choice (number): ").strip()
            if choice in options:
                return choice
            print("  ✗ Invalid choice. Please try again.")
        except KeyboardInterrupt:
            print("\n\n✗ Cancelled")
            sys.exit(0)


def create_sample_dataset():
    """Create sample videos for testing"""
    print_header("Creating Sample Videos")
    
    print("\n  This will create 2 sample videos (real & fake) for testing.")
    print("  Location: DATA/Real/ and DATA/Fake/")
    
    os.system('python ../scripts/download_deepfake_dataset.py --create-samples')


def download_youtube_videos():
    """Download from YouTube"""
    print_header("Download from YouTube")
    
    print("\n  YouTube has many face videos (interviews, vlogs, presentations)")
    print("  for creating a real video dataset.")
    
    print_option(1, "Single video download")
    print_option(2, "Batch download from URL list")
    print_option(3, "Go back")
    
    choice = get_choice(['1', '2', '3'])
    
    if choice == '1':
        url = input("\n  → Enter YouTube URL: ").strip()
        category = input("  → Category (real/fake) [default: real]: ").strip() or 'real'
        
        os.system(f'python ../scripts/download_youtube_videos.py --url "{url}" --category {category}')
    
    elif choice == '2':
        print("\n  1. Create a text file (e.g., urls.txt) with YouTube URLs, one per line")
        print("  2. Run: python ../scripts/download_youtube_videos.py --file urls.txt --category real")
        input("\n  Press Enter when ready...")


def manual_collection():
    """Manual dataset collection instructions"""
    print_header("Manual Dataset Collection")
    
    instructions = """
  Option A: Use Existing Video Files
  ──────────────────────────────────
  1. Find real videos:
     - Record yourself or friends (with permission)
     - Download from Creative Commons sources
     - Use royalty-free video sites

  2. Find deepfake/manipulated videos:
     - Search for deepfake demonstrations (research purposes)
     - Use faces from movies/actors (transformations)
     - Create your own with FaceSwap or DeepFaceLab

  3. Organize:
     Place videos in:
       DATA/Real/      (real person videos)
       DATA/Fake/      (deepfake videos)

  Option B: Download Datasets
  ──────────────────────────
  Large public datasets available:

  1. FaceForensics++
     - http://kaldir.vc.in.tum.de/faceforensics/
     - Request access on website
     - High-quality deepfakes

  2. DFDC (Deepfake Detection Challenge)
     - https://www.deepfakedetectionchallenge.org/
     - Download dataset with instructions

  3. Celeb-DF
     - https://www.cs.albany.edu/~lsw/celeb-df.html
     - ~590 real videos, ~5639 deepfakes

  Option C: Create Your Own Deepfakes
  ───────────────────────────────────
  Tools to create deepfakes (research only):

  1. FaceSwap
     - https://github.com/deepfakes/faceswap
     - Face-swapping deepfakes

  2. DeepFaceLab
     - https://github.com/iperov/DeepFaceLab
     - Professional-grade deepfakes

  3. Roop
     - https://github.com/s0md3v/roop
     - Simple one-click deepfake creation
"""
    
    print(instructions)
    input("\n  Press Enter to continue...")


def check_dataset():
    """Check current dataset status"""
    print_header("Dataset Status")
    
    data_dir = Path("../../DATA")
    real_dir = data_dir / "Real"
    fake_dir = data_dir / "Fake"
    
    if not data_dir.exists():
        print("\n  ✗ DATA directory not found")
        print(f"  Create at: {data_dir.absolute()}")
        return
    
    real_videos = list(real_dir.glob('*.mp4')) + list(real_dir.glob('*.mov')) + list(real_dir.glob('*.avi'))
    fake_videos = list(fake_dir.glob('*.mp4')) + list(fake_dir.glob('*.mov')) + list(fake_dir.glob('*.avi'))
    
    print(f"\n  📁 Real Videos: {len(real_videos)}")
    for v in real_videos[:10]:
        size = v.stat().st_size / (1024*1024)
        print(f"     - {v.name} ({size:.1f} MB)")
    if len(real_videos) > 10:
        print(f"     ... and {len(real_videos) - 10} more")
    
    print(f"\n  📁 Fake Videos: {len(fake_videos)}")
    for v in fake_videos[:10]:
        size = v.stat().st_size / (1024*1024)
        print(f"     - {v.name} ({size:.1f} MB)")
    if len(fake_videos) > 10:
        print(f"     ... and {len(fake_videos) - 10} more")
    
    total_size = sum(v.stat().st_size for v in real_videos + fake_videos) / (1024*1024)
    
    print(f"\n  📊 Total Videos: {len(real_videos) + len(fake_videos)}")
    print(f"  💾 Total Size: {total_size:.1f} MB")
    
    ready = len(real_videos) > 0 and len(fake_videos) > 0
    print(f"\n  {'✓' if ready else '✗'} Ready for Training: {ready}")
    
    if ready:
        print("\n  Next step - train the model:")
        print("  $ cd Backend/pytorch")
        print("  $ python train_video.py --config config_video.yaml")


def show_tips():
    """Show dataset collection tips"""
    print_header("Dataset Collection Tips")
    
    tips = """
  📊 Dataset Quality
  ──────────────────
  ✓ Balance: Equal number of real and fake videos
  ✓ Diversity: Different people, angles, lighting, expressions
  ✓ Quality: Minimum 480p, preferably 720p or higher
  ✓ Duration: 2-30 seconds per video (longer is better)
  ✓ Format: MP4, MOV, or AVI (any standard video format)

  🎯 Getting Started
  ──────────────────
  Minimum needed to start training:
    - 5 real videos
    - 5 fake/manipulated videos

  Recommended for good results:
    - 20-50 real videos
    - 20-50 fake/deepfake videos

  Production quality:
    - 100+ real videos
    - 100+ fake/deepfake videos

  🚀 Quick Start (< 15 minutes)
  ────────────────────────────
  1. Run: python dataset_setup.py
  2. Select "Option 1: Create Sample Videos"
  3. Wait for sample generation (creates 2 test videos)
  4. Run training: python train_video.py --config config_video.yaml

  📥 Medium Setup (1-2 hours)
  ──────────────────────────
  1. Download 10-20 real face videos from YouTube or other sources
  2. Download 10-20 deepfake videos from public datasets
  3. Place in DATA/Real/ and DATA/Fake/
  4. Train model with good accuracy

  🎬 Professional Setup (Multiple hours)
  ─────────────────────────────────────
  1. Collect or download 100+ real videos
  2. Generate or download 100+ deepfake videos
  3. Verify video quality and consistency
  4. Train model for excellent accuracy

  ⚠️  Legal & Ethical Considerations
  ──────────────────────────────────
  - Only use videos you have permission to use
  - Respect copyright and privacy
  - For research purposes only
  - Follow local laws regarding deepfakes
  - Do not create non-consensual deepfakes
  - Properly attribute and cite sources
"""
    
    print(tips)
    input("\n  Press Enter to continue...")


def main():
    """Main menu"""
    print("\n")
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║       DEEPFAKE DATASET SETUP WIZARD                             ║")
    print("║       Interactive guide for collecting training data            ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    
    while True:
        print_header("Main Menu")
        
        print_option(1, "✓ Create Sample Videos (Quick Test - 2 minutes)")
        print_option(2, "📥 Download from YouTube")
        print_option(3, "📋 Manual Collection Guide")
        print_option(4, "📊 Check Dataset Status")
        print_option(5, "💡 Show Tips & Best Practices")
        print_option(6, "❌ Exit")
        
        choice = get_choice(['1', '2', '3', '4', '5', '6'])
        
        if choice == '1':
            create_sample_dataset()
            input("\n  Press Enter to return to menu...")
        elif choice == '2':
            download_youtube_videos()
        elif choice == '3':
            manual_collection()
        elif choice == '4':
            check_dataset()
            input("\n  Press Enter to return to menu...")
        elif choice == '5':
            show_tips()
        elif choice == '6':
            print("\n  ✓ Goodbye!")
            print("\n  Remember: Once you have videos in DATA/Real/ and DATA/Fake/")
            print("  Run: python train_video.py --config config_video.yaml\n")
            break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✗ Cancelled")
        sys.exit(0)
