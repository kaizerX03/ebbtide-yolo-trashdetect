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
    model_path = "/home/pi/ebbtide-yolo-trashdetect/best_ncnn_model/ebb_ncnn_model"
    if os.path.exists(model_path):
        print(f"✅ YOLO model found at {model_path}")
    else:
        print(f"❌ YOLO model not found at {model_path}")
        print("   Please place your trained model file in this location")
    
    print("\n🎉 Setup test complete!")
    print("\nNext steps:")
    print("1. Place your YOLO model file at /home/pi/ebbtide-yolo-trashdetect/best_ncnn_model/ebb_ncnn_model")
    print("2. Reboot your Raspberry Pi: sudo reboot")
    print("3. Run the detection script: ./start_trash_collector.sh")

if __name__ == "__main__":
    main()
