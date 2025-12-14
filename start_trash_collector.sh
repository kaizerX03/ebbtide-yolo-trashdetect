#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate virtual environment
source "$SCRIPT_DIR/ebb_env/bin/activate"

# Change to project directory
cd "$SCRIPT_DIR"

# Check if model exists
if [ ! -d "$SCRIPT_DIR/best_ncnn_model" ]; then
    echo "❌ YOLO model not found at $SCRIPT_DIR/best_ncnn_model"
    echo "Please place your trained model directory in the correct location"
    exit 1
fi

# Run the detection script
echo "🔍 Starting YOLO trash detection..."
python3 yolo_detect.py
