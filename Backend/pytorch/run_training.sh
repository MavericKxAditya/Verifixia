#!/bin/bash
# Activate virtual environment
source ../venv/bin/activate

# Run training
python3 train_improved.py --config config.yaml
