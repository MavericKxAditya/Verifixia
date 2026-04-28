#!/bin/bash

# Verifixia AI Backend Runner Script

echo "🚀 Starting Verifixia AI Backend..."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check if virtual environment exists, create if not
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies (excluding PyTorch for now)
echo "📥 Installing basic dependencies..."
pip install Flask Flask-CORS python-dotenv Pillow

echo "⚠️  Note: PyTorch not installed. Backend will run in mock mode."
echo "   To enable real AI predictions, install PyTorch manually:"
echo "   pip install torch torchvision torchaudio"

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p uploads

# Create basic model if it doesn't exist (skip if PyTorch not available)
if command -v python3 &> /dev/null && python3 -c "import torch" &> /dev/null; then
    if [ ! -f "../models/xception_deepfake.pth" ]; then
        echo "🤖 Creating basic model..."
        python3 create_model.py
    fi
else
    echo "⚠️  PyTorch not available - skipping model creation"
fi

# Run the Flask app
echo "🌐 Starting Flask server on http://localhost:5000"
python3 app.py