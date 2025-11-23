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

## Operation Modes

### NORMAL Mode (Default)
This is the standard autonomous trash collection mode that works with GPS waypoint missions.

**Prerequisites:**
- GPS lock acquired
- Waypoint mission loaded in Mission Planner
- Pixhawk flight mode set to AUTO

**How it works:**
1. The boat follows the waypoint mission in AUTO mode
2. When trash is detected, the system automatically switches to GUIDED mode
3. The boat enters autonomous navigation phases:
   - **ALIGN**: Rotates to center the trash in view using differential steering
   - **ADVANCE**: Moves forward toward the trash while maintaining alignment
   - **COLLECT**: Stops motors and waits for collection confirmation (if dual-camera enabled)
4. After collection or timeout, the boat returns to AUTO mode and resumes the waypoint mission

**Starting NORMAL Mode:**
1. Ensure `test_mode.test_nav_mode: false` in `config/detection_config.yaml`
2. Upload waypoint mission to Pixhawk via Mission Planner
3. Switch Pixhawk to AUTO mode using RC transmitter
4. Run the script: `python yolo_detect.py`
5. The boat will follow waypoints and automatically navigate to detected trash

### TEST_NAV Mode
This is a standalone testing mode that allows you to test trash navigation without a GPS mission.

**Prerequisites:**
- GPS lock acquired (for HOLD mode stability)
- No waypoint mission required
- Pixhawk flight mode set to HOLD

**How it works:**
1. The boat maintains GPS position in HOLD mode
2. When trash is detected, a 3-second buffer countdown starts automatically
3. The system temporarily switches to MANUAL mode and takes control
4. The boat navigates through the same ALIGN → ADVANCE → COLLECT phases
5. After collection or timeout, the system returns to HOLD mode
6. The boat automatically resumes GPS position holding

**Starting TEST_NAV Mode:**
1. Set `test_mode.test_nav_mode: true` in `config/detection_config.yaml`
2. Run the script: `python yolo_detect.py`
3. Follow this operational sequence:

```
Step 1: Arm the boat         → Press 'a'
Step 2: Set mode to HOLD     → Press 'm', then type 'HOLD' or 'h'
Step 3: Enable motor control → Press 'e' (if not already enabled)
Step 4: Boat stays still     → Waiting for trash detection...
Step 5: Trash detected!      → 3-second buffer countdown starts
Step 6: Auto approach begins → HOLD → MANUAL (system takes control)
Step 7: Navigation phases    → ALIGN → ADVANCE → COLLECT
Step 8: Returns to HOLD      → Boat resumes stationary GPS hold
```

The boat will automatically handle the mode switching and navigation. You just need to position trash in front of the camera and the system will navigate to it, then return to HOLD mode.

## Keyboard Controls

### General Controls
- **'q'**: Quit the program
- **'s'**: Pause/Resume detection
- **'p'**: Save screenshot of current frame
- **'e'**: Enable/disable automatic motor control on detection

### Pixhawk Connection & Diagnostics
- **'c'**: Reconnect to Pixhawk (if connection lost)
- **'d'**: Run diagnostic port scan
- **'h'**: Show Pixhawk command help

### Flight Mode & Motor Controls
- **'m'**: Manual flight mode selection (type mode name in terminal)
- **'a'**: Arm vehicle (motors ready, sets neutral position)
- **'f'**: Disarm vehicle (motors off)
- **'z'**: Return to Launch (RTL mode)

### Motor Testing (Vehicle must be armed first)
- **'r'**: Test right motor (channel 1) - runs at 1894 PWM for 5 seconds
- **'l'**: Test left motor (channel 3) - runs at 1894 PWM for 5 seconds

**⚠️ Safety Warning**: Ensure propellers are clear and boat is secured before testing motors!

## Configuration

Edit `config/detection_config.yaml` to customize:
- Detection thresholds and confidence levels
- Motor PWM values (forward thrust, turn gain, neutral positions)
- Navigation parameters (collection distance, arrival thresholds)
- Timeout values (detection buffer, lost target, idle brake)
- Dual-camera settings for collection confirmation
- Test mode settings (TEST_NAV mode toggle)

## Hardware Setup

### Required Components:
- Raspberry Pi 4 (or newer)
- Raspberry Pi Camera Module (primary detection camera)
- Pixhawk flight controller running ArduRover firmware
- GPS module (for HOLD and AUTO modes)
- Electronic Speed Controllers (ESCs) calibrated for:
  - Right motor (channel 1): Neutral at 1335 PWM
  - Left motor (channel 3): Neutral at 1500 PWM
- Optional: USB camera for collection confirmation

### Connections:
- Pixhawk UART to Raspberry Pi GPIO (default: `/dev/ttyAMA0` at 57600 baud)
- Pi Camera to CSI port
- Optional USB camera to USB port

## Motor Calibration

The system uses calibrated neutral PWM values specific to your ESCs:
- **Right motor neutral**: 1335 PWM (channel 1)
- **Left motor neutral**: 1500 PWM (channel 3)

These values were determined during ESC calibration and represent the PWM where each motor stops spinning. Different neutral values are normal due to ESC manufacturing variance.

**To recalibrate** (if you replace an ESC):
1. Arm the vehicle using 'a'
2. Test each motor individually using 'r' and 'l'
3. Find the PWM value where the motor just stops spinning
4. Update the values in `yolo_detect.py` in the `set_neutral_motors()` function
