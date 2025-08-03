#!/bin/bash

# Raspberry Pi Setup Script for YOLO Trash Detection Project
# This script will install all necessary dependencies for the project

echo "=================================================="
echo "🚀 Setting up Raspberry Pi for YOLO Trash Detection"
echo "=================================================="

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo "❌ Please do not run this script as root/sudo"
    echo "The script will prompt for sudo when needed"
    exit 1
fi

# Check if filesystem is read-only and make it writable
echo "🔧 Checking filesystem status..."
if mount | grep -q "/ .*ro,"; then
    echo "📝 Filesystem is read-only, making it temporarily writable..."
    sudo mount -o remount,rw /
    FILESYSTEM_WAS_READONLY=true
    echo "✅ Filesystem is now writable"
else
    echo "✅ Filesystem is already writable"
    FILESYSTEM_WAS_READONLY=false
fi

# Update system packages
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install system dependencies
echo "🔧 Installing system dependencies..."
sudo apt install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    cmake \
    build-essential \
    pkg-config \
    libopencv-dev \
    libopencv-contrib-dev \
    libatlas-base-dev \
    libjpeg-dev \
    libtiff5-dev \
    libpng-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libfontconfig1-dev \
    libcairo2-dev \
    libgdk-pixbuf2.0-dev \
    libpango1.0-dev \
    libgtk2.0-dev \
    libgtk-3-dev \
    libhdf5-dev \
    libhdf5-serial-dev \
    libhdf5-103 \
    libqtgui4 \
    libqtwebkit4 \
    libqt4-test \
    python3-pyqt5 \
    git \
    curl \
    wget

# Install Pi Camera dependencies
echo "📷 Installing Pi Camera dependencies..."
sudo apt install -y \
    python3-picamera2 \
    libcamera-apps \
    libcamera-dev

# Enable camera interface
echo "🔧 Enabling camera interface..."
#sudo raspi-config nonint do_camera 0

# Enable I2C and SPI (sometimes needed for sensors)
echo "🔧 Enabling I2C and SPI..."
#sudo raspi-config nonint do_i2c 0
#sudo raspi-config nonint do_spi 0

# Enable serial port for Pixhawk communication
echo "🔧 Configuring serial port for Pixhawk..."
#sudo raspi-config nonint do_serial 2  # Enable serial, disable console

# Add user to dialout group for serial communication
echo "👤 Adding user to dialout group..."
sudo usermod -a -G dialout $USER

VENV_DIR="/home/pi/ebbtide-yolo-trashdetect/ebb_env"
echo "🐍 Creating Python virtual environment at $VENV_DIR with system site packages..."
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv --system-site-packages $VENV_DIR
    echo "✅ Virtual environment created at $VENV_DIR (system site packages enabled)"
else
    echo "✅ Virtual environment already exists at $VENV_DIR"
fi


# Activate virtual environment and prepare for package installation
echo "📦 Activating Python virtual environment..."
source $VENV_DIR/bin/activate

# Upgrade pip
pip install --upgrade pip

# Uninstall all existing Python packages in the venv
echo "🧹 Uninstalling all existing Python packages in the virtual environment..."
pip freeze | xargs -r pip uninstall -y
echo "✅ All existing Python packages uninstalled."


# Install core dependencies with pinned versions
echo "Installing core Python packages..."
pip install \
    ultralytics==8.3.7 \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1 \
    opencv-python \
    opencv-contrib-python \
    numpy \
    PyYAML \
    pillow \
    matplotlib \
    scipy \
    scikit-learn \
    pandas

# Install Picamera2 in virtual environment
echo "📷 Installing Picamera2 in virtual environment..."
pip install picamera2

# Install DroneKit and related packages
echo "🚁 Installing DroneKit and MAVLink packages..."
pip install \
    dronekit \
    pymavlink \
    dronekit-sitl

MODEL_DIR="$VENV_DIR"
echo "📁 Creating model directory at $MODEL_DIR..."
mkdir -p $MODEL_DIR

# Set proper permissions for UART devices
echo "🔧 Setting up UART permissions..."
sudo chmod 666 /dev/ttyAMA0 2>/dev/null || echo "ttyAMA0 not available"
sudo chmod 666 /dev/ttyACM0 2>/dev/null || echo "ttyACM0 not available"
sudo chmod 666 /dev/ttyUSB0 2>/dev/null || echo "ttyUSB0 not available"

echo "🚀 Creating startup script..."
cat > /home/pi/ebbtide-yolo-trashdetect/start_trash_collector.sh << 'EOF'
#!/bin/bash

# Activate virtual environment
source /home/pi/ebbtide-yolo-trashdetect/ebb_env/bin/activate
source /home/pi/ebbtide-yolo-trashdetect/ebb_env/bin/activate

# Change to project directory
cd /home/pi/ebbtide-yolo-trashdetect

# Check if model exists
if [ ! -f "/home/pi/ebbtide-yolo-trashdetect/ebb_env/ebb_ncnn_model" ]; then
    echo "❌ YOLO model not found at /home/pi/ebbtide-yolo-trashdetect/ebb_env/ebb_ncnn_model"
    echo "Please place your trained model file in the correct location"
    exit 1
fi

# Run the detection script
echo "🔍 Starting YOLO trash detection..."
python3 yolo_detect.py
EOF

chmod +x /home/pi/ebbtide-yolo-trashdetect/start_trash_collector.sh

echo "🧪 Creating test script..."
cat > /home/pi/ebbtide-yolo-trashdetect/test_setup.py << 'EOF'
#!/usr/bin/env python3
"""
Test script to verify all dependencies are properly installed
"""

import sys
import importlib

def test_import(module_name, description=""):
    try:
        importlib.import_module(module_name)
        print(f"✅ {module_name} - {description}")
        return True
    except ImportError as e:
        print(f"❌ {module_name} - {description}: {e}")
        return False

def test_camera():
    """Test camera functionality"""
    print("\n📷 Testing camera functionality...")
    
    # Test OpenCV
    try:
        import cv2
        print(f"✅ OpenCV version: {cv2.__version__}")
        
        # Test USB camera
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✅ USB camera accessible")
            cap.release()
        else:
            print("⚠️  USB camera not accessible (may not be connected)")
    except Exception as e:
        print(f"❌ OpenCV camera test failed: {e}")
    
    # Test Pi Camera
    try:
        from picamera2 import Picamera2
        picam2 = Picamera2()
        print("✅ Pi Camera accessible")
    except Exception as e:
        print(f"⚠️  Pi Camera test failed: {e}")

def test_serial():
    """Test serial port access for Pixhawk"""
    print("\n🔌 Testing serial port access...")
    import os
    
    serial_devices = ['/dev/ttyAMA0', '/dev/ttyACM0', '/dev/ttyUSB0', '/dev/serial0']
    for device in serial_devices:
        if os.path.exists(device):
            try:
                with open(device, 'r') as f:
                    print(f"✅ {device} - accessible")
            except PermissionError:
                print(f"❌ {device} - permission denied")
            except Exception as e:
                print(f"⚠️  {device} - {e}")
        else:
            print(f"⚠️  {device} - does not exist")

def main():
    print("🧪 Testing YOLO Trash Detection Setup")
    print("=" * 40)
    
    # Test core dependencies
    print("\n📦 Testing Python packages...")
    test_import("cv2", "OpenCV")
    test_import("numpy", "NumPy")
    test_import("yaml", "PyYAML")
    test_import("ultralytics", "Ultralytics YOLO")
    test_import("torch", "PyTorch")
    test_import("torchvision", "TorchVision")
    test_import("picamera2", "Pi Camera 2")
    test_import("dronekit", "DroneKit")
    test_import("pymavlink", "PyMAVLink")
    
    # Test camera
    test_camera()
    
    # Test serial
    test_serial()
    
    # Check model file
    print("\n🤖 Checking YOLO model...")
    model_path = "/home/pi/ebbtide-yolo-trashdetect/ebb_env/ebb_ncnn_model"
    if os.path.exists(model_path):
        print(f"✅ YOLO model found at {model_path}")
    else:
        print(f"❌ YOLO model not found at {model_path}")
        print("   Please place your trained model file in this location")
    
    print("\n🎉 Setup test complete!")
    print("\nNext steps:")
    print("1. Place your YOLO model file at /home/pi/ebbtide-yolo-trashdetect/ebb_env/ebb_ncnn_model")
    print("2. Reboot your Raspberry Pi: sudo reboot")
    print("3. Run the detection script: ./start_trash_collector.sh")

if __name__ == "__main__":
    main()
EOF

chmod +x /home/pi/ebbtide-yolo-trashdetect/test_setup.py

# Deactivate virtual environment
deactivate

# Restore read-only filesystem if it was originally read-only
if [ "$FILESYSTEM_WAS_READONLY" = true ]; then
    echo "🔒 Restoring read-only filesystem..."
    sudo mount -o remount,ro /
    echo "✅ Filesystem is now read-only again"
fi

echo ""
echo "=================================================="
echo "🎉 Setup Complete!"
echo "=================================================="
echo ""
echo "What was installed:"
echo "✅ System packages (OpenCV, build tools, etc.)"
echo "✅ Python virtual environment at /home/pi/ebbtide-yolo-trashdetect/ebb_env (with system site packages)"
echo "✅ Python packages (ultralytics 8.3.7, PyTorch 2.5.1, OpenCV, DroneKit, etc.)"
echo "✅ Pi Camera support"
echo "✅ Serial port configuration for Pixhawk"
echo "✅ Startup script: start_trash_collector.sh"
echo "✅ Test script: test_setup.py"
echo ""
echo "Next steps:"
echo "1. Reboot your Raspberry Pi: sudo reboot"
echo "2. Test the installation: python3 test_setup.py"
echo "3. Place your YOLO model at: /home/pi/ebbtide-yolo-trashdetect/best_ncnn_model/ebb_ncnn_model"
echo "4. Run the project: ./start_trash_collector.sh"
echo ""
if [ "$FILESYSTEM_WAS_READONLY" = true ]; then
    echo "🔒 FILESYSTEM PROTECTION: Your system has been restored to read-only mode"
    echo "   This protects against corruption and unauthorized changes."
    echo ""
fi
echo "⚠️  IMPORTANT: You need to reboot for all changes to take effect!"
echo ""
