# YOLO Trash Detection with Pixhawk Integration

This project integrates YOLO object detection with Pixhawk motor control via DroneKit on a Raspberry Pi. The system can:

1. Detect objects using YOLO
2. Control bidirectional motors via Pixhawk
3. Activate motors when trash is detected
4. Display real-time feedback and diagnostics

## Features
- Object detection with configurable confidence threshold
- Distance estimation
- Bidirectional motor control
- Visual feedback with onscreen indicators
- Detailed diagnostic logging
- Motor activation on detection with configurable timing

## Usage
Run the main script:
```
python yolo_detect.py
```

## Controls
- 'q': Quit
- 's': Pause/Resume
- 'p': Save screenshot
- 'c': Reconnect to Pixhawk
- 'd': Run diagnostic port scan
- 'h': Show Pixhawk command help
- 'e': Enable/disable motor control on detection
- 'r': Test right motor
- 'l': Test left motor
