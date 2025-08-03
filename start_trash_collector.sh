#!/bin/bash

# Activate virtual environment
source /home/pi/yolo_env/bin/activate

# Change to project directory
cd /home/pi/ebbtide-yolo-trashdetect

# Check if model exists
if [ ! -f "/home/pi/yolo_env/ebb_ncnn_model" ]; then
    echo "❌ YOLO model not found at /home/pi/yolo_env/ebb_ncnn_model"
    echo "Please place your trained model file in the correct location"
    exit 1
fi

# Run the detection script
echo "🔍 Starting YOLO trash detection..."
python3 yolo_detect.py
