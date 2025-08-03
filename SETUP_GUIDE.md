# Raspberry Pi Setup Guide for YOLO Trash Detection

This guide will help you set up your Raspberry Pi from scratch for the YOLO trash detection project.

## Quick Setup (Recommended)

1. **Run the automated setup script:**
   ```bash
   chmod +x setup_raspberry_pi.sh
   ./setup_raspberry_pi.sh
   ```

2. **Reboot your Raspberry Pi:**
   ```bash
   sudo reboot
   ```

3. **Test the installation:**
   ```bash
   source /home/pi/yolo_env/bin/activate
   python3 test_setup.py
   ```

4. **Place your YOLO model file:**
   - Copy your trained model file to: `/home/pi/yolo_env/ebb_ncnn_model`
   - Or update the path in `config/detection_config.yaml`

5. **Run the detection system:**
   ```bash
   ./start_trash_collector.sh
   ```

## Manual Setup (Alternative)

If you prefer to install packages manually:

### 1. System Dependencies
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-pip python3-venv cmake build-essential
sudo apt install -y libopencv-dev libatlas-base-dev libjpeg-dev
sudo apt install -y python3-picamera2 libcamera-apps
```

### 2. Enable Hardware Interfaces
```bash
sudo raspi-config nonint do_camera 0    # Enable camera
sudo raspi-config nonint do_serial 2    # Enable serial for Pixhawk
sudo raspi-config nonint do_i2c 0       # Enable I2C
sudo raspi-config nonint do_spi 0       # Enable SPI
```

### 3. Create Virtual Environment
```bash
python3 -m venv /home/pi/yolo_env
source /home/pi/yolo_env/bin/activate
pip install --upgrade pip
```

### 4. Install Python Packages
```bash
pip install -r requirements.txt
```

### 5. Setup Permissions
```bash
sudo usermod -a -G dialout $USER
sudo chmod 666 /dev/ttyAMA0  # For Pixhawk communication
```

## Configuration

### Camera Setup
- **USB Camera**: Set `source: "usb0"` in `config/detection_config.yaml`
- **Pi Camera**: Set `source: "picamera0"` in `config/detection_config.yaml`

### Pixhawk Setup
- Connect via USB or UART
- Update connection port in `config/detection_config.yaml`
- Common ports: `/dev/ttyACM0` (USB), `/dev/ttyAMA0` (UART)

### Model Setup
- Place your trained YOLO model at: `/home/pi/yolo_env/ebb_ncnn_model`
- Or update the path in `config/detection_config.yaml`

## Troubleshooting

### Common Issues

1. **"Permission denied" for serial ports:**
   ```bash
   sudo usermod -a -G dialout $USER
   sudo reboot
   ```

2. **Pi Camera not detected:**
   ```bash
   sudo raspi-config nonint do_camera 0
   sudo reboot
   ```

3. **DroneKit import errors:**
   ```bash
   python3 patch_dronekit.py
   ```

4. **OpenCV installation issues:**
   ```bash
   pip install opencv-python opencv-contrib-python
   ```

### Hardware Requirements
- Raspberry Pi 4 (recommended) or Pi 3B+
- MicroSD card (32GB+ recommended)
- Pi Camera or USB camera
- Pixhawk flight controller (optional)

### Performance Tips
- Use a fast MicroSD card (Class 10 or better)
- Ensure adequate cooling for sustained operation
- Consider using lower resolution for better FPS
- Use GPU acceleration when available

## Testing the Setup

Run the test script to verify everything is working:
```bash
source /home/pi/yolo_env/bin/activate
python3 test_setup.py
```

This will check:
- Python package installations
- Camera functionality
- Serial port access
- Model file presence

## Running the System

Once everything is set up:
```bash
./start_trash_collector.sh
```

Controls:
- `q`: Quit the application
- `s`: Pause/unpause
- `p`: Save current frame
- `c`: Calibrate (in calibration mode)

## Project Structure

```
ebbtide-yolo-trashdetect/
├── config/
│   └── detection_config.yaml    # Main configuration
├── calibration_images/          # Camera calibration images
├── calibration_output/          # Calibration results
├── yolo_detect.py              # Main detection script
├── calibrate_with_object.py    # Camera calibration tool
├── patch_dronekit.py           # DroneKit compatibility fixes
├── start_trash_collector.sh    # Startup script
├── setup_raspberry_pi.sh       # Setup automation script
├── test_setup.py               # Installation test script
└── requirements.txt            # Python dependencies
```
