import os
import sys
import time
import yaml
import cv2
import numpy as np
import logging
import json
from dataclasses import dataclass
from ultralytics import YOLO

# Make Picamera2 import optional
PICAMERA_AVAILABLE = False
try:
    from picamera2 import Picamera2
    PICAMERA_AVAILABLE = True
except ImportError:
    print("Warning: Picamera2 module not available. USB camera will still work.")

# Add DroneKit import for Pixhawk connection
# Fix for collections.MutableMapping issue in Python 3.10+
import collections.abc
if not hasattr(collections, 'MutableMapping'):
    collections.MutableMapping = collections.abc.MutableMapping

try:
    from dronekit import connect, VehicleMode, APIException, LocationGlobal, LocationGlobalRelative
    DRONEKIT_AVAILABLE = True
except ImportError:
    print("Warning: DroneKit module not available. Pixhawk connection won't work.")
    DRONEKIT_AVAILABLE = False
except Exception as e:
    print(f"Warning: DroneKit import error: {e}")
    DRONEKIT_AVAILABLE = False

# ---------------------------
# Configuration and Setup
# ---------------------------

"""YOLO trash detection script (config-driven, no CLI args)."""

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger("yolo_detect")

def load_config():
    path = os.path.join(os.path.dirname(__file__), 'config', 'detection_config.yaml')
    try:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        log.error(f'Could not load configuration file: {e}')
        sys.exit(1)

config = load_config()
model_path = config['model']['path']
img_source = config['camera']['source']
min_thresh = float(config['model']['confidence_threshold'])
try:
    resW, resH = map(int, config['camera']['resolution'].split('x'))
except ValueError:
    log.error("Invalid resolution format in config (expected WxH)")
    sys.exit(1)
record = bool(config.get('recording', {}).get('enabled', False))
distance_estimation_config = config.get('distance_estimation', {})
focal_length_px = distance_estimation_config.get('focal_length_px')
known_object_height_mm = distance_estimation_config.get('known_object_height_mm')
detection_required_frames = int(config.get('detection_logic', {}).get('consecutive_frames', 2))

# Collection camera configuration
collection_cam_config = config.get('collection_camera', {})
collection_cam_enabled = bool(collection_cam_config.get('enabled', False))
collection_cam_source = collection_cam_config.get('source', 'usb0')
collection_cam_res = collection_cam_config.get('resolution', '320x240')
collection_line_percent = float(collection_cam_config.get('collection_line_percent', 0.7))
collection_cam_fps = int(collection_cam_config.get('fps', 10))
enable_mode_switch = bool(config.get('detection_logic', {}).get('enable_mode_switch', True))
detection_consecutive_frames = 0
pixhawk_settings = config.get('pixhawk', {})
pixhawk_connection = pixhawk_settings.get('connection', '/dev/ttyAMA0')
pixhawk_baud = int(pixhawk_settings.get('baud_rate', 57600))
pixhawk_timeout = int(pixhawk_settings.get('timeout', 30))
pixhawk_enabled = bool(pixhawk_settings.get('control_enabled', False))

# Navigation (hybrid diversion) configuration
nav_cfg = config.get('navigation', {})
nav_mode = nav_cfg.get('mode', 'direct_control')
forward_thrust_pwm = int(nav_cfg.get('forward_thrust_pwm', 1600))
turn_gain_pwm = int(nav_cfg.get('turn_gain_pwm', 500))
center_dead_zone = float(nav_cfg.get('center_dead_zone', 0.15))
collection_distance_m = float(nav_cfg.get('collection_distance_m', 1.0))
lost_target_timeout_s = float(nav_cfg.get('lost_target_timeout_s', 3.0))
max_approach_time_s = float(nav_cfg.get('max_approach_time_s', 30.0))
distance_filter_alpha = float(nav_cfg.get('distance_filter_alpha', 0.3))
camera_fov_deg = float(nav_cfg.get('camera_fov_deg', 60.0))
arrival_bbox_min_px = int(nav_cfg.get('arrival_bbox_min_px', 120))
nav_logging_enabled = bool(nav_cfg.get('logging_enabled', True))
nav_log_file = nav_cfg.get('log_file', 'nav_events.log')
suppression_timeout_s = float(nav_cfg.get('suppression_timeout_s', 90.0))
align_hold = bool(nav_cfg.get('align_hold', False))
idle_brake_s = float(nav_cfg.get('idle_brake_s', 0.5))

# Test mode flags
test_mode_cfg = config.get('test_mode', {})
simulate_navigation = bool(test_mode_cfg.get('simulate_navigation', False))
test_nav_mode = bool(test_mode_cfg.get('test_nav_mode', False))
detection_buffer_time = float(test_mode_cfg.get('detection_buffer_time', 3.0))

@dataclass
class ApproachState:
    active: bool = False
    start_time: float = 0.0
    last_seen_time: float = 0.0
    phase: str = 'idle'  # idle | align | advance | collect | abort
    smoothed_distance_m: float = None
    target_bearing: float = None
    center_x: int = None
    center_y: int = None
    center_error: float = None  # normalized horizontal error [-1..1]
    bbox_height_px: int = None
    last_distance_m: float = None
    last_thrust_log_time: float = 0.0
    completed: bool = False
    last_steering_zone: str = None  # Track zone transitions: 'left', 'center', 'right'

approach_state = ApproachState()

# Post-arrival detection suppression state
detection_suppressed = False
suppression_target_index = None
suppression_start_time = 0.0

# Collection tracking
total_trash_collected = 0
last_collection_time = 0.0
collection_cooldown = 3.0  # Prevent double-counting same trash

# Detection buffer (for TEST_NAV mode)
detection_buffer_start_time = 0.0
detection_buffering = False

def log_nav_event(event_type, **fields):
    if not nav_logging_enabled:
        return
    
    # Helper to convert numpy types to native python types
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return obj

    payload = {
        'ts': time.time(),
        'event': event_type,
        'phase': approach_state.phase,
        'center_error': convert_types(approach_state.center_error),
        'distance_m': convert_types(approach_state.last_distance_m),
        'bbox_height_px': convert_types(approach_state.bbox_height_px),
    }
    
    # Add extra fields with conversion
    for k, v in fields.items():
        payload[k] = convert_types(v)
        
    try:
        with open(nav_log_file, 'a') as f:
            f.write(json.dumps(payload) + '\n')
    except Exception as e:
        print(f"[NAV][LOG] Failed to write log: {e}")

if not os.path.exists(model_path):
    log.error('Model path is invalid or model was not found.')
    sys.exit(0)

# ---------------------------
# Model and Camera Initialization
# ---------------------------

model = YOLO(model_path, task='detect')
labels = ['Human','NonTrash', 'Trash']  # Replace with your actual classes


if 'usb' in img_source:
    source_type = 'usb'
    usb_idx = int(img_source[3:])
    cap = cv2.VideoCapture(usb_idx)
    cap.set(3, resW)
    cap.set(4, resH)
elif 'picamera' in img_source:
    if not PICAMERA_AVAILABLE:
        print("ERROR: Picamera2 module not available but 'picamera' source was specified.")
        print("Please install picamera2 or change source to 'usb0' in config.")
        sys.exit(1)
    source_type = 'picamera'
    cap = Picamera2()
    config_cam = cap.create_video_configuration(
        main={"format": 'RGB888', "size": (resW, resH)}
    )
    cap.configure(config_cam)
    cap.start()
    time.sleep(1.0)
else:
    print('Invalid source. Use "usb0" for webcam or "picamera0" for Pi Camera.')
    sys.exit(0)

if record:
    record_name = 'demo1.avi'
    record_fps = 30
    recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (resW, resH))

bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), 
              (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

# ---------------------------
# Collection Camera Setup (Optional)
# ---------------------------
collection_cap = None
collection_resW, collection_resH = 320, 240

if collection_cam_enabled:
    try:
        collection_resW, collection_resH = map(int, collection_cam_res.split('x'))
        
        if 'usb' in collection_cam_source:
            collection_idx = int(collection_cam_source[3:])
            collection_cap = cv2.VideoCapture(collection_idx)
            
            if collection_cap.isOpened():
                collection_cap.set(3, collection_resW)
                collection_cap.set(4, collection_resH)
                collection_cap.set(cv2.CAP_PROP_FPS, collection_cam_fps)
                print(f"Collection camera initialized: {collection_cam_source} at {collection_cam_res}")
            else:
                print(f"WARNING: Could not open collection camera {collection_cam_source}")
                collection_cap = None
                collection_cam_enabled = False
        else:
            print("WARNING: Collection camera only supports USB cameras (usb0, usb1)")
            collection_cam_enabled = False
            
    except Exception as e:
        print(f"WARNING: Failed to initialize collection camera: {e}")
        collection_cap = None
        collection_cam_enabled = False
else:
    print("Collection camera disabled in config")

collection_line_y = int(collection_resH * collection_line_percent)

# ---------------------------
# Pixhawk Integration
# ---------------------------
vehicle = None

def set_neutral_motors():
    """Set motors to calibrated neutral values (safety helper)."""
    global vehicle
    if vehicle is None:
        return
    try:
        vehicle.channels.overrides['1'] = 1335  # Steering neutral
        vehicle.channels.overrides['3'] = 1500  # Throttle neutral
    except Exception as e:
        log.warning(f"Neutral motor set failed: {e}")

def connect_pixhawk(connection_string='/dev/ttyAMA0', baud_rate=57600, timeout=30):
    """Establish a connection to the Pixhawk via UART. Default is ttyAMA0 which is the standard UART port."""
    global vehicle
    
    if not DRONEKIT_AVAILABLE:
        print("ERROR: DroneKit not available. Cannot connect to Pixhawk.")
        return False
    
    # First check if the port exists
    if not os.path.exists(connection_string):
        print(f"ERROR: Port {connection_string} does not exist.")
        return False
        
    # Check permissions on port
    try:
        os.access(connection_string, os.R_OK | os.W_OK)
    except:
        print(f"WARNING: May not have permission to access {connection_string}.")
        print(f"You might need to run: sudo chmod 666 {connection_string}")
    
    print(f"Connecting to vehicle on: {connection_string}, baud={baud_rate}, timeout={timeout}s")
    try:
        # Connect to the Vehicle with a timeout to avoid hanging
        # Try a more direct connection method
        vehicle = connect(connection_string, baud=baud_rate, wait_ready=False, heartbeat_timeout=timeout)
        print("Initial connection established...")
        
        # Now wait for parameters to download
        print("Waiting for vehicle to initialize...")
        vehicle.wait_ready(True, timeout=timeout)
        
        print(f"Connected to vehicle!")
        print(f" > System status: {vehicle.system_status.state}")
        print(f" > Mode: {vehicle.mode.name}")
        print(f" > GPS: {vehicle.gps_0.fix_type}")
        print(f" > Battery: {vehicle.battery.voltage}V")
        
        # Ensure all motor outputs are at correct neutral position upon connection
        print("Setting all motors to neutral position...")
        # Clear any existing RC overrides
        vehicle.channels.overrides = {}
        # Explicitly set motor channels to their specific neutral positions
        vehicle.channels.overrides['1'] = 1335  # Right motor neutral (1335)
        vehicle.channels.overrides['3'] = 1500  # Left motor neutral (1500)
        
        return True
    except APIException as e:
        print(f"Connection timed out: {e}")
        print("Try increasing the timeout or check the baud rate.")
        return False
    except OSError as e:
        print(f"OS Error: {e}")
        print("Check that the port exists and you have the right permissions.")
        return False
    except Exception as e:
        print(f"Connection failed: {e}")
        return False
        
def get_vehicle_status():
    """Get current status of the connected vehicle."""
    if vehicle is None:
        return "Not connected"
    
    try:
        status = {
            "armed": vehicle.armed,
            "mode": vehicle.mode.name,
            "system_status": vehicle.system_status.state,
            "gps": vehicle.gps_0.fix_type,
            "battery": f"{vehicle.battery.voltage:.1f}V"
        }
        return status
    except Exception as e:
        print(f"Error getting vehicle status: {e}")
        return "Error getting status"

def diagnostic_port_scan():
    """Scan available serial ports and check if they might be a Pixhawk"""
    print("\nScanning for available serial ports...")
    
    # List of common serial port patterns on Raspberry Pi
    possible_ports = [
        '/dev/ttyAMA0', '/dev/ttyAMA1',  # Hardware UART
        '/dev/ttyS0',                    # Serial 0
        '/dev/serial0',                  # Symlink to primary serial port
        '/dev/ttyUSB0', '/dev/ttyUSB1'   # USB-to-Serial adapters
    ]
    
    found_ports = []
    for port in possible_ports:
        if os.path.exists(port):
            # Check if we have read/write permissions
            if os.access(port, os.R_OK | os.W_OK):
                perms = "READ/WRITE OK"
            else:
                perms = "PERMISSION DENIED"
            found_ports.append(f"{port} - {perms}")
    
    if found_ports:
        print("Found the following serial ports:")
        for port in found_ports:
            print(f"  {port}")
        print("\nIf you're having connection issues, try:")
        print("1. Ensure the UART is enabled in raspi-config")
        print("2. Fix permissions: sudo chmod 666 /dev/ttyAMA0 (or appropriate port)")
        print("3. Try different baud rates (common: 57600, 115200)")
    else:
        print("No serial ports found! Check your system configuration.")
    
    return found_ports

# ---------------------------
# Pixhawk Control Commands
# ---------------------------

def set_flight_mode(mode_name):
    """Change the flight mode of the vehicle."""
    global vehicle
    if vehicle is None:
        print("ERROR: Cannot set mode - no active connection to Pixhawk")
        return False
    
    try:
        print(f"Changing flight mode to {mode_name}...")
        vehicle.mode = VehicleMode(mode_name)
        
        # Wait for mode change to take effect
        for _ in range(5):  # Try for a few seconds
            if vehicle.mode.name == mode_name:
                print(f"Flight mode changed to: {vehicle.mode.name}")
                return True
            time.sleep(0.5)
        
        print(f"Failed to change mode to {mode_name}. Current mode: {vehicle.mode.name}")
        return False
    except Exception as e:
        print(f"Error changing flight mode: {e}")
        return False

def arm_disarm(arm_command):
    """Arm or disarm the vehicle."""
    global vehicle
    if vehicle is None:
        print("ERROR: Cannot arm/disarm - no active connection to Pixhawk")
        return False
    
    try:
        if arm_command and not vehicle.armed:
            # First check if armable
            if not vehicle.is_armable:
                print("WARNING: Vehicle reports 'not armable' (GPS/EKF/Safety Switch).")
                print(f"Current status: {vehicle.system_status.state}")
                print("Attempting to arm anyway (Force Arm)...")
                # return False  <-- Commented out to allow force arming
            
            print("Arming motors...")
            vehicle.armed = True
            
            # Wait for arming to take effect
            for _ in range(10):  # Try for a few seconds
                if vehicle.armed:
                    print("Vehicle armed!")
                    # Immediately set all motors to their specific neutral positions
                    print("Setting all motors to neutral position...")
                    set_neutral_motors()
                    return True
                time.sleep(0.5)
            
            print("Failed to arm vehicle!")
            return False
            
        elif not arm_command and vehicle.armed:
            print("Disarming motors...")
            vehicle.armed = False
            
            # Wait for disarming to take effect
            for _ in range(5):  # Try for a few seconds
                if not vehicle.armed:
                    print("Vehicle disarmed!")
                    return True
                time.sleep(0.5)
            
            print("Failed to disarm vehicle!")
            return False
        
        else:
            print(f"Vehicle already {'armed' if vehicle.armed else 'disarmed'}")
            return True
            
    except Exception as e:
        print(f"Error during arm/disarm command: {e}")
        return False
    
def takeoff(target_altitude):
    """Take off to a specified altitude (in meters)."""
    global vehicle
    if vehicle is None:
        print("ERROR: Cannot takeoff - no active connection to Pixhawk")
        return False
    
    try:
        if not vehicle.armed:
            print("Vehicle not armed! Cannot takeoff.")
            return False
            
        print(f"Taking off to {target_altitude}m altitude...")
        
        # Use simple_takeoff
        vehicle.simple_takeoff(target_altitude)
        
        print("Takeoff command sent! Vehicle will climb to target altitude.")
        return True
    except Exception as e:
        print(f"Error during takeoff: {e}")
        return False
        
def clear_all_motor_overrides():
    """
    Clear all RC overrides, returning control to RC transmitter or stopping motors.
    This is a safety function to ensure motors don't keep running.
    
    Returns:
        Success status (bool)
    """
    global vehicle
    if vehicle is None:
        print("ERROR: Cannot clear overrides - no active connection to Pixhawk")
        return False
    
    try:
        # Clear all channel overrides
        vehicle.channels.overrides = {}
        print("All RC channel overrides cleared")
        return True
    except Exception as e:
        print(f"Error clearing RC overrides: {e}")
        return False

def control_motors(right_motor_value=None, left_motor_value=None, should_clear=False):
    """
    Control the motors connected to Pixhawk outputs 1 (right) and 3 (left).
    
    Args:
        right_motor_value: PWM value for the right motor (channel 1), 1000-2000
                          1000 = off, 1500 = mid, 2000 = full throttle
                          If None, this motor is not changed
        left_motor_value: PWM value for the left motor (channel 3), 1000-2000
                         1000 = off, 1500 = mid, 2000 = full throttle
                         If None, this motor is not changed
        should_clear: Whether to clear all overrides after setting values.
                     Only use this when you want to remove all overrides.
                         
    Returns:
        Success status (bool)
    """
    global vehicle, motor_control_enabled
    if vehicle is None:
        print("ERROR: Cannot control motors - no active connection to Pixhawk")
        return False
    
    # Only perform the safety check if motor control is disabled
    # or if we're doing explicit motor tests (keyboard test)
    if not motor_control_enabled:
        # Allow explicit testing through keyboard commands
        caller = sys._getframe(1).f_code.co_name
        is_keyboard_test = caller == "<module>"
        
        # If not an explicit test and trying to set motors above minimum
        if not is_keyboard_test and (
            (right_motor_value is not None and right_motor_value > 1000) or 
            (left_motor_value is not None and left_motor_value > 1000)):
            print("WARNING: Attempt to set motors above minimum while motor control is disabled")
            print("Overriding to minimum throttle for safety")
            right_motor_value = 1000 if right_motor_value is not None else None
            left_motor_value = 1000 if left_motor_value is not None else None
    
    try:
        # Validate and apply right motor value (channel 1)
        if right_motor_value is not None:
            # Ensure value is within valid PWM range
            right_motor_value = max(1000, min(2000, right_motor_value))
            vehicle.channels.overrides['1'] = right_motor_value
            print(f"Right motor (channel 1) set to {right_motor_value}")
            
        # Validate and apply left motor value (channel 3)
        if left_motor_value is not None:
            # Ensure value is within valid PWM range
            left_motor_value = max(1000, min(2000, left_motor_value))
            vehicle.channels.overrides['3'] = left_motor_value
            print(f"Left motor (channel 3) set to {left_motor_value}")
            
        # Only clear all overrides if explicitly requested
        if should_clear:
            print("Clearing all RC channel overrides")
            vehicle.channels.overrides = {}
            
        return True
    except Exception as e:
        print(f"Error controlling motors: {e}")
        return False

def check_collection_confirmation(detections, frame):
    """Check if trash crossed the collection line on conveyor camera.
    
    Args:
        detections: YOLO detection results
        frame: Camera frame
        
    Returns:
        bool: True if trash crossed collection line
    """
    global total_trash_collected, last_collection_time
    
    # Draw collection line
    cv2.line(frame, (0, collection_line_y), (collection_resW, collection_line_y), 
             (0, 255, 0), 2)
    cv2.putText(frame, 'COLLECTION LINE', (10, collection_line_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    current_time = time.time()
    
    # Check cooldown to prevent double-counting
    if current_time - last_collection_time < collection_cooldown:
        return False
    
    # Check if any trash bbox crosses the line
    for detection in detections:
        conf = detection.conf.item()
        if conf > min_thresh:
            xyxy = detection.xyxy.cpu().numpy().squeeze()
            xmin, ymin, xmax, ymax = xyxy.astype(int)
            bbox_bottom = ymax
            
            # Draw bbox on collection camera
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 255), 2)
            
            # Check if bottom of bbox crossed the collection line
            if bbox_bottom >= collection_line_y:
                total_trash_collected += 1
                last_collection_time = current_time
                
                # Visual feedback
                cv2.putText(frame, 'COLLECTED!', (xmin, ymin - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                print(f"[COLLECTION] Trash collected! Total: {total_trash_collected}")
                log_nav_event('trash_collected', total_count=total_trash_collected)
                
                return True
    
    return False

def show_command_help():
    """Display available commands for the Pixhawk."""
    print("\n--- Pixhawk Commands ---")
    print("  'm': Change flight mode")
    print("  'a': Arm motors")
    print("  'f': Disarm motors")
    print("  'h': Show this help menu")
    # Differential steering: Ch3=Throttle (fwd/rev), Ch1=Steering (left/right)
    print("  'l': Steer LEFT test")
    print("  'k': Reverse THROTTLE test")
    print("  'r': Steer RIGHT test")
    print("  't': Forward THROTTLE test")
    print("  'e': Toggle auto navigation thrust (detection diversion ON/OFF)")
    print("  'z': Return to Launch (RTL)")
    print("  'b': Reboot Pixhawk")
    print("------------------------")
    print("Current keyboard controls:")
    print("  'q': Quit")
    print("  's': Pause/Resume")
    print("  'p': Save screenshot")
    print("  'c': Reconnect to Pixhawk")
    print("  'd': Run diagnostic port scan")

# ---------------------------
# Detection and Frame Processing
# ---------------------------

# Global flags to enable/disable motor control based on detections
motor_control_enabled = False
motor_active = False  # Flag to track if motors are currently active
last_detection_time = 0
motor_run_time = 5.0  # How long to run motors after detection (seconds)
detection_cooldown = 3.0  # Minimum time between motor activations (seconds)
# Additional flag to simplify motor control logic
motor_start_time = 0  # Time when motors were activated

# Mode switching control
mode_switch_on_detection = True  # Flag to enable automatic mode switching on detection (always on)
original_mode = None  # Store original mode to switch back if needed
mode_switch_time = 0  # When we switched modes
mode_switch_duration = 10.0  # How long to stay in GUIDED mode after last detection (seconds)
in_detection_mode = False  # Flag to indicate we're in detection handling mode
last_detection_confirmed_time = time.time()  # Initialize with current time to prevent errors

def handle_detections(detections, frame):
    """
    Handle motor control and mode switching based on detected objects.
    
    Args:
        detections: List of object detections
        frame: The current video frame
    """
    global vehicle, motor_control_enabled, last_detection_time, motor_active, motor_start_time
    global mode_switch_on_detection, original_mode, mode_switch_time, in_detection_mode
    # Suppression state globals (needed because we assign to them below)
    global detection_suppressed, suppression_target_index, suppression_start_time
    
    # If vehicle isn't connected, do nothing
    if vehicle is None or not vehicle.armed:
        return
    
    # Get current time
    current_time = time.time()

    # Post-arrival suppression: ignore detections until next waypoint index advances or timeout
    if detection_suppressed:
        # Draw suppression overlay
        sup_text = "SUPPRESSING DETECTIONS - RETURNING TO MISSION"
        ts_sup = cv2.getTextSize(sup_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.putText(frame, sup_text,
                    (frame.shape[1]//2 - ts_sup[0]//2, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        # Check waypoint advancement if in AUTO
        if vehicle.mode.name == 'AUTO' and suppression_target_index is not None:
            try:
                current_wp = vehicle.commands.next
                if current_wp != suppression_target_index:
                    # Waypoint advanced -> end suppression
                    detection_suppressed = False
                    log_nav_event('suppression_end', reason='waypoint_advanced', from_index=suppression_target_index, new_index=current_wp)
                    print(f"[NAV] Suppression ended (waypoint advanced {suppression_target_index} -> {current_wp})")
                elif (current_time - suppression_start_time) > suppression_timeout_s:
                    # Timeout fallback
                    detection_suppressed = False
                    log_nav_event('suppression_end', reason='timeout')
                    print("[NAV] Suppression timeout reached - re-enabling detections.")
            except Exception as e:
                print(f"[NAV] Suppression waypoint check error: {e}")
        # Skip rest of detection handling while suppressed
        return
    
    # Handle mode switching automatically based on detections
    # This always runs because mode_switch_on_detection is always True
    
    # Raw detection evaluation + debounce
    has_detections_raw = False
    # Determine most-centered detection and compute normalized horizontal error
    frame_h, frame_w = frame.shape[:2]
    frame_cx = frame_w / 2.0
    frame_cy = frame_h / 2.0
    most_centered = None
    most_centered_dist = None
    for detection in detections:
        conf = detection.conf.item()
        if conf > min_thresh:
            # Mark presence of any raw detections
            has_detections_raw = True
            xyxy = detection.xyxy.cpu().numpy().squeeze()
            xmin, ymin, xmax, ymax = xyxy.astype(int)
            cx = (xmin + xmax) / 2.0
            cy = (ymin + ymax) / 2.0
            dist = ((cx - frame_cx) ** 2 + (cy - frame_cy) ** 2) ** 0.5
            if most_centered_dist is None or dist < most_centered_dist:
                most_centered_dist = dist
                most_centered = (int(cx), int(cy), xmin, ymin, xmax, ymax)

    # If we found a most-centered detection compute horizontal error
    center_error = None
    if most_centered:
        mcx, mcy, xmin, ymin, xmax, ymax = most_centered
        center_error = (mcx - frame_cx) / frame_cx  # range approx [-1..1]
        # Visual marker (distinct from existing MOST CENTERED marker if any)
        cv2.putText(frame, f"ERR {center_error:+.2f}", (mcx+10, mcy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
        bbox_height_px = ymax - ymin
        est_distance_m = None
        if focal_length_px and known_object_height_mm and bbox_height_px > 0:
            distance_mm = (known_object_height_mm * focal_length_px) / bbox_height_px
            est_distance_m = distance_mm / 1000.0
            # Smooth distance via EMA
            if approach_state.last_distance_m is None:
                approach_state.last_distance_m = est_distance_m
            else:
                approach_state.last_distance_m = (distance_filter_alpha * est_distance_m +
                                                  (1 - distance_filter_alpha) * approach_state.last_distance_m)

        # Update approach state tracking if active
        if approach_state.active:
            approach_state.center_x = mcx
            approach_state.center_y = mcy
            approach_state.center_error = center_error
            approach_state.bbox_height_px = bbox_height_px
    global detection_consecutive_frames, last_detection_confirmed_time
    if has_detections_raw:
        detection_consecutive_frames += 1
    else:
        detection_consecutive_frames = 0
    has_detections = detection_consecutive_frames >= detection_required_frames
    if has_detections:
        last_detection_confirmed_time = current_time
    
    current_mode = vehicle.mode.name
    
    # TEST_NAV MODE: Boat stays in HOLD until trash detected, then approaches
    if test_nav_mode:
        global detection_buffer_start_time, detection_buffering
        
        # If we have detections and we're in HOLD, start buffer countdown
        if has_detections and current_mode == 'HOLD' and not in_detection_mode and not detection_buffering:
            detection_buffering = True
            detection_buffer_start_time = current_time
            print(f"\n[TEST_NAV] Trash detected! Starting {detection_buffer_time}s buffer countdown...")
            print(f"[TEST_NAV] Place trash in front of boat now!")
        
        # Show buffer countdown on screen
        if detection_buffering:
            buffer_elapsed = current_time - detection_buffer_start_time
            buffer_remaining = detection_buffer_time - buffer_elapsed
            
            if buffer_remaining > 0:
                # Display countdown
                countdown_text = f"DETECTION BUFFER: {buffer_remaining:.1f}s"
                textsize = cv2.getTextSize(countdown_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)[0]
                cv2.putText(frame, countdown_text,
                           (frame.shape[1]//2 - textsize[0]//2, frame.shape[0]//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
                cv2.putText(frame, "Place trash in front of boat",
                           (frame.shape[1]//2 - 200, frame.shape[0]//2 + 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Keep motors neutral during buffer
                set_neutral_motors()
            else:
                # Buffer complete - switch to MANUAL and start approach
                detection_buffering = False
                original_mode = 'HOLD'
                print(f"\n[TEST_NAV] Buffer complete! Switching from HOLD to MANUAL for approach")
                
                try:
                    vehicle.mode = VehicleMode('MANUAL')
                    in_detection_mode = True
                    mode_switch_time = current_time
                    set_neutral_motors()
                    
                    # Show mode switch notification
                    text = "TEST_NAV: HOLD → MANUAL (APPROACHING)"
                    textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    cv2.putText(frame, text, 
                              (frame.shape[1]//2 - textsize[0]//2, frame.shape[0] - 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                except Exception as e:
                    print(f"Error switching mode: {e}")
                
                # Activate approach state
                if not approach_state.active:
                    approach_state.active = True
                    approach_state.start_time = current_time
                    approach_state.last_seen_time = current_time
                    approach_state.phase = 'align'
                    approach_state.center_error = center_error
                    if most_centered:
                        approach_state.center_x = most_centered[0]
                        approach_state.center_y = most_centered[1]
                        approach_state.bbox_height_px = (most_centered[5] - most_centered[3]) if len(most_centered) >= 6 else None
                    print("[TEST_NAV] Approach state activated")
                    log_nav_event('test_nav_approach_start', mode=current_mode)
        
        # If detection lost during buffer, cancel buffer
        if detection_buffering and not has_detections:
            print("[TEST_NAV] Detection lost during buffer - cancelling")
            detection_buffering = False
        
        # After no detections for duration, return to HOLD instead of AUTO
        if in_detection_mode:
            time_since_last_detection = current_time - last_detection_confirmed_time
            if time_since_last_detection >= mode_switch_duration and original_mode == 'HOLD':
                print(f"\n[TEST_NAV] No detections for {mode_switch_duration}s. Returning to HOLD")
                try:
                    vehicle.mode = VehicleMode('HOLD')
                    vehicle.channels.overrides = {}
                    in_detection_mode = False
                    original_mode = None
                    detection_buffering = False  # Reset buffer state
                    print("[TEST_NAV] Returned to HOLD - boat stationary")
                except Exception as e:
                    print(f"Error returning to HOLD: {e}")
        
        # Skip the normal AUTO mode logic below when in TEST_NAV mode
        if current_mode != 'AUTO':
            # Continue to navigation logic at end of function
            pass
        else:
            # If somehow in AUTO during TEST_NAV, warn user
            cv2.putText(frame, "WARNING: TEST_NAV requires HOLD mode, not AUTO", 
                       (10, frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # NORMAL MODE: If we have detections and we're not already in MANUAL mode, switch to MANUAL
    # OR if we are already in MANUAL and want to activate tracking (Bench Test)
    elif has_detections and (current_mode == 'AUTO' or (current_mode == 'MANUAL' and motor_control_enabled)) and not in_detection_mode:
        
        if current_mode == 'AUTO':
            # Store the original mode so we can switch back later
            original_mode = current_mode
            
            print(f"\nDetection triggered auto mode switch! Current mode: {current_mode}")
            print(f"Switching to MANUAL mode for object handling")
            
            # Switch to MANUAL mode
            try:
                vehicle.mode = VehicleMode('MANUAL')
                in_detection_mode = True
                mode_switch_time = current_time
                
                # Safety first - stop the vehicle by setting motors to neutral
                set_neutral_motors()
                
                # Show mode switch notification (centered)
                text = f"SWITCHING MODES: {current_mode} → MANUAL"
                textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.putText(frame, text, 
                          (frame.shape[1]//2 - textsize[0]//2, frame.shape[0] - 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            except Exception as e:
                print(f"Error switching mode: {e}")
        elif current_mode == 'MANUAL':
            # Already in MANUAL, just activate tracking logic
            # We don't set in_detection_mode=True because we don't want to auto-switch back to anything
            pass

        # Hybrid navigation activation (skeleton)
        if pixhawk_enabled and nav_mode == 'direct_control' and not approach_state.active:
            approach_state.active = True
            approach_state.start_time = current_time
            approach_state.last_seen_time = current_time
            approach_state.phase = 'advance'  # Start directly in advance mode for simple steering
            print("[NAV] ApproachState activated (phase=advance). Steering towards target.")
            # Initialize error values if available
            approach_state.center_error = center_error
            if most_centered:
                approach_state.center_x = most_centered[0]
                approach_state.center_y = most_centered[1]
                approach_state.bbox_height_px = (most_centered[5] - most_centered[3]) if len(most_centered) >= 6 else None
            log_nav_event('approach_start', mode=current_mode)
    
    # If we're in detection mode, show status and check for switching back
    if in_detection_mode:
        # If we have no detections for the duration period, switch back to original mode
        time_since_last_detection = current_time - last_detection_confirmed_time
        if time_since_last_detection >= mode_switch_duration:
            # Switch back to original mode
            if original_mode is not None:
                print(f"\nNo detections for {mode_switch_duration} seconds.")
                print(f"Switching back to {original_mode} mode")
                try:
                    vehicle.mode = VehicleMode(original_mode)
                    in_detection_mode = False
                    original_mode = None
                    
                    # Now that we're switching back to AUTO mode, display a clear message
                    print("Mode switch complete - Resuming motion in AUTO mode")
                    
                    # Make sure we're not setting overrides anymore
                    # This allows the AUTO mode to take full control
                    vehicle.channels.overrides = {}
                    
                    # Draw a message indicating motion is resuming (moved to bottom)
                    text = "RETURNING TO AUTO - RESUMING MOTION"
                    textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    cv2.putText(frame, text, 
                              (frame.shape[1]//2 - textsize[0]//2, frame.shape[0] - 120),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                except Exception as e:
                    print(f"Error switching back to original mode: {e}")
        
        # Show mode information on screen (centered)
        text = f"DETECTION MODE: MANUAL"
        textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.putText(frame, text, (frame.shape[1]//2 - textsize[0]//2, frame.shape[0] - 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        if has_detections:
            text = f"Object detected - holding in MANUAL mode"
            textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.putText(frame, text, 
                       (frame.shape[1]//2 - textsize[0]//2, frame.shape[0] - 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # Basic placeholder: show NAV active status if diversion started
            if approach_state.active:
                remaining = max_approach_time_s - (current_time - approach_state.start_time)
                nav_text = f"NAV divert phase={approach_state.phase} t_left={remaining:.1f}s"
                ts2 = cv2.getTextSize(nav_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.putText(frame, nav_text,
                            (frame.shape[1]//2 - ts2[0]//2, frame.shape[0] - 145),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
                # Show current horizontal error if available
                if approach_state.center_error is not None:
                    err_text = f"center_error={approach_state.center_error:+.2f}"
                    ts3 = cv2.getTextSize(err_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    cv2.putText(frame, err_text,
                                (frame.shape[1]//2 - ts3[0]//2, frame.shape[0] - 165),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,200), 1)
        else:
            remaining_time = mode_switch_duration - time_since_last_detection
            text = f"No detections - returning to {original_mode} in: {remaining_time:.1f}s"
            textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.putText(frame, text, 
                       (frame.shape[1]//2 - textsize[0]//2, frame.shape[0] - 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # If motor control is disabled, still allow mode switching but don't control motors
    if not motor_control_enabled:
        return

    # Navigation diversion logic
    if approach_state.active:
        # Abort: lost target
        if (current_time - last_detection_confirmed_time) > lost_target_timeout_s:
            print("[NAV] Lost target - aborting diversion.")
            approach_state.active = False
            approach_state.phase = 'abort'
            set_neutral_motors()
            log_nav_event('abort', reason='lost_target')
            return
        # Abort: max time
        if (current_time - approach_state.start_time) > max_approach_time_s:
            print("[NAV] Max approach time exceeded - aborting diversion.")
            approach_state.active = False
            approach_state.phase = 'abort'
            set_neutral_motors()
            log_nav_event('abort', reason='timeout')
            return

        # Require center error to proceed
        if approach_state.center_error is None:
            # Still waiting for a detection with center data
            return

        # Simplified Navigation Logic (Bang-Bang Controller)
        # Always in 'advance' phase as per user request
        approach_state.phase = 'advance'
        
        # Determine steering based on position relative to dead zone
        # center_error is [-1.0 (Left) ... 0.0 (Center) ... +1.0 (Right)]
        
        # Differential steering system:
        # Channel 3 = Throttle (forward/reverse for both motors)
        # Channel 1 = Steering (differential between motors)
        STEERING_NEUTRAL = 1335  # Channel 1 neutral
        THROTTLE_NEUTRAL = 1500  # Channel 3 neutral
        
        # Determine current steering zone
        current_zone = None
        
        # Calculate steering and throttle
        if approach_state.center_error < -center_dead_zone:
            # Target is LEFT -> Steer LEFT
            current_zone = 'left'
            steering_pwm = STEERING_NEUTRAL - turn_gain_pwm  # Steer left
            throttle_pwm = forward_thrust_pwm  # Move forward
            print(f"[NAV] TURNING LEFT: Steering={steering_pwm}, Throttle={throttle_pwm}, error={approach_state.center_error:.2f}")
        elif approach_state.center_error > center_dead_zone:
            # Target is RIGHT -> Steer RIGHT
            current_zone = 'right'
            steering_pwm = STEERING_NEUTRAL + turn_gain_pwm  # Steer right
            throttle_pwm = forward_thrust_pwm  # Move forward
            print(f"[NAV] TURNING RIGHT: Steering={steering_pwm}, Throttle={throttle_pwm}, error={approach_state.center_error:.2f}")
        else:
            # Target CENTERED -> Go STRAIGHT FORWARD
            current_zone = 'center'
            steering_pwm = STEERING_NEUTRAL  # Straight
            throttle_pwm = 1850  # Move forward with good speed for demo
            print(f"[NAV] STRAIGHT FORWARD: Steering={steering_pwm}, Throttle={throttle_pwm}, error={approach_state.center_error:.2f}")

        # Detect zone transitions and force clear PWM update
        zone_changed = (approach_state.last_steering_zone != current_zone)
        if zone_changed and approach_state.last_steering_zone is not None:
            print(f"[NAV] ZONE TRANSITION: {approach_state.last_steering_zone} -> {current_zone} - Forcing PWM update")
            # Clear overrides momentarily to ensure clean transition
            if vehicle and vehicle.armed and not simulate_navigation:
                try:
                    vehicle.channels.overrides['1'] = STEERING_NEUTRAL
                    vehicle.channels.overrides['3'] = THROTTLE_NEUTRAL
                    time.sleep(0.05)  # Brief pause to ensure command is processed
                except Exception as e:
                    print(f"[NAV] Zone transition clear error: {e}")
        
        approach_state.last_steering_zone = current_zone

        # Clip to safe bounds (1000-2000)
        def clip_pwm(val):
            return max(1000, min(2000, val))

        steering_pwm = clip_pwm(steering_pwm)
        throttle_pwm = clip_pwm(throttle_pwm)
        print(f"[NAV] After clipping: Steering={steering_pwm}, Throttle={throttle_pwm}")

        # Idle brake: if detections temporarily lost, stop forward thrust but keep phase

        no_detect_elapsed = current_time - last_detection_confirmed_time
        if no_detect_elapsed > idle_brake_s and approach_state.phase in ('align','advance'):
            # Set to neutral for both channels
            steering_pwm = 1335  # Channel 1 neutral
            throttle_pwm = 1500  # Channel 3 neutral

        # Apply or simulate
        if simulate_navigation or vehicle is None or not vehicle.armed:
            print(f"[NAV][SIM] phase={approach_state.phase} err={approach_state.center_error:+.2f} steering={steering_pwm} throttle={throttle_pwm}")
        else:
            try:
                vehicle.channels.overrides['1'] = steering_pwm  # Steering
                vehicle.channels.overrides['3'] = throttle_pwm  # Throttle
            except Exception as e:
                print(f"[NAV] Override error: {e}")
        # Periodic thrust logging (1s cadence)
        if current_time - approach_state.last_thrust_log_time >= 1.0:
            log_nav_event('thrust', steering_pwm=steering_pwm, throttle_pwm=throttle_pwm, err=approach_state.center_error)
            approach_state.last_thrust_log_time = current_time

        # Overlay current thrust values
        thrust_text = f"Steer:{steering_pwm} Thr:{throttle_pwm} phase={approach_state.phase}"
        ts_thr = cv2.getTextSize(thrust_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.putText(frame, thrust_text,
                    (frame.shape[1]//2 - ts_thr[0]//2, frame.shape[0] - 185),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,180,255), 1)

        # ========== ARRIVAL LOGIC COMMENTED OUT ==========
        # User requested: continue navigation without stopping when arrived
        # Only stop when no trash is detected (handled by lost_target_timeout_s)
        
        # # Arrival condition: either distance below threshold or bbox height above arrival threshold
        # arrived = False
        # if approach_state.last_distance_m is not None and approach_state.last_distance_m <= collection_distance_m:
        #     arrived = True
        # elif approach_state.bbox_height_px is not None and approach_state.bbox_height_px >= arrival_bbox_min_px:
        #     arrived = True

        # if arrived and approach_state.phase in ('align','advance'):
        #     print(f"[NAV] Arrival reached (distance={approach_state.last_distance_m}, bbox_h={approach_state.bbox_height_px}). Holding and completing diversion.")
        #     set_neutral_motors()
        #     approach_state.phase = 'collect'
        #     # Display arrival message
        #     arr_text = "ARRIVED - NEUTRAL HOLD"
        #     ts_arr = cv2.getTextSize(arr_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        #     cv2.putText(frame, arr_text,
        #                 (frame.shape[1]//2 - ts_arr[0]//2, frame.shape[0] - 205),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        #     # Mark diversion complete; allow mode logic to revert via existing timer or immediate if desired
        #     approach_state.active = False
        #     # Optionally shorten mode_switch_duration on arrival (future improvement)
        #     approach_state.completed = True
        #     log_nav_event('arrival', distance=approach_state.last_distance_m, bbox_h=approach_state.bbox_height_px)
        #     # Immediate AUTO resume if we switched from AUTO earlier
        #     if in_detection_mode and original_mode is not None:
        #         try:
        #             vehicle.mode = VehicleMode(original_mode)
        #             vehicle.channels.overrides = {}
        #             in_detection_mode = False
        #             log_nav_event('resume_auto', prev_mode='MANUAL')
        #             print("[NAV] Switched back to AUTO immediately after arrival.")
        #             # Begin suppression until next waypoint advances
        #             try:
        #                 suppression_target_index = vehicle.commands.next
        #                 detection_suppressed = True
        #                 suppression_start_time = time.time()
        #                 log_nav_event('suppression_start', target_index=suppression_target_index)
        #                 print(f"[NAV] Detection suppression started (waiting for waypoint index to advance from {suppression_target_index}).")
        #             except Exception as e:
        #                 print(f"[NAV] Could not start suppression (command index unavailable): {e}")
        #             original_mode = None
        #         except Exception as e:
        #             print(f"[NAV] Error resuming AUTO: {e}")
        #     return
        
        # ========== END ARRIVAL LOGIC ==========

        # Continue navigating as long as trash is detected
        # Will only stop when lost_target_timeout_s is exceeded (no detections)
        return
    
    # Get current time
    current_time = time.time()
    
    # STAGE 1: MOTORS ACTIVE - Motors are currently paused
    # REMOVED: Conflicting "pause motors" logic to allow active navigation
    
    # STAGE 2: COOLDOWN - Wait cooldown period after motors stop
    # REMOVED: Cooldown logic to allow continuous tracking
    
    # STAGE 3: DETECTION - Check for objects to activate motors
    # REMOVED: Old "stop on detection" logic. 
    # Navigation is now handled by the 'approach_state' block above.
    
    return

def process_frame(frame):
    """Run detection and annotate frame."""
    global last_detection_time, motor_run_time, motor_active
    results = model(frame, verbose=False)
    detections = results[0].boxes
    
    # Handle detections for motor control
    handle_detections(detections, frame)
    
    # We don't need the extra safety check here anymore as the handle_detections 
    # function handles the motor timing correctly
    
    object_count = 0
    most_centered_idx = None
    min_center_dist = None
    frame_center = (resW // 2, resH // 2)
    centers = []
    for i, detection in enumerate(detections):
        # Get bounding box coordinates
        xyxy = detection.xyxy.cpu().numpy().squeeze()
        xmin, ymin, xmax, ymax = xyxy.astype(int)

        # Get class info and confidence
        classidx = int(detection.cls.item())
        classname = labels[classidx]
        conf = detection.conf.item()

        # Draw box if confidence exceeds threshold
        if conf > min_thresh:
            color = bbox_colors[classidx % 10]
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), color, 2)

            # Draw label
            label = f'{classname}: {int(conf*100)}%'

            # Estimate and display distance if parameters are available
            if focal_length_px and known_object_height_mm:
                object_height_px = ymax - ymin
                if object_height_px > 0:
                    distance_mm = (known_object_height_mm * focal_length_px) / object_height_px
                    distance_m = distance_mm / 1000
                    label += f' {distance_m:.2f}m'

            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), 
                         (xmin+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED)
            cv2.putText(frame, label, (xmin, label_ymin-7), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            object_count += 1

            # Calculate center of bbox
            center_x = (xmin + xmax) // 2
            center_y = (ymin + ymax) // 2
            centers.append((center_x, center_y))

                    # Calculate distance to frame center
            dist = np.hypot(center_x - frame_center[0], center_y - frame_center[1])
            if min_center_dist is None or dist < min_center_dist:
                min_center_dist = dist
                most_centered_idx = i

    # Highlight most centered object
    if most_centered_idx is not None:
        cx, cy = centers[most_centered_idx]
        cv2.drawMarker(frame, (cx, cy), (0,255,0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
        cv2.putText(frame, 'MOST CENTERED', (cx-60, cy-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    
    # Simplified safety check
    # Only run if motors are NOT active to avoid interference with active motor control
    current_time = time.time()  # Get current time for safety check
    if vehicle is not None and vehicle.armed and not motor_active:
        # Only check every 5 seconds to avoid flooding console
        if int(current_time) % 5 == 0:
            try:
                # Only check when motors should be idle (not in active motor control period)
                if not motor_control_enabled:
                    right_motor = vehicle.channels['1']
                    left_motor = vehicle.channels['3']
                    
                    # Check if motors are significantly off from their specific neutral positions
                    if abs(right_motor - 1335) > 50 or abs(left_motor - 1500) > 50:
                        print(f"\nSAFETY CHECK: Motors off neutral when they should be idle!")
                        print(f"Values: Right={right_motor}, Left={left_motor}")
                        print("Setting motors to neutral positions...")
                        
                        # Set motors to neutral with direct dictionary assignment
                        set_neutral_motors()
            except Exception as e:
                print(f"Safety check error: {e}")
        
    return frame, object_count

# ---------------------------
# Main Loop
# ---------------------------

# Print keyboard controls
print("\nKeyboard Controls:")
print("  'q': Quit")
print("  's': Pause/Resume")
print("  'p': Save screenshot")
print("  'c': Reconnect to Pixhawk (if disconnected)")
print("  'd': Run diagnostic port scan")
print("  'h': Show Pixhawk command help")
print("  'e': Toggle auto navigation thrust (diversion ON/OFF)")
print("  'r': Test right motor (channel 1)")
print("  'l': Test left motor (channel 3)")
print("  'a': Arm motors")
print("  'f': Disarm motors")
print("  'm': Change flight mode")
print("  'b': Reboot Pixhawk")
print("  'z': Return to Launch (RTL)")

if test_nav_mode:
    print("\n=== TEST_NAV MODE ACTIVE ===")
    print("Instructions for testing trash navigation:")
    print("  1. Arm the boat ('a' key)")
    print("  2. Set mode to HOLD ('m' key, then select HOLD)")
    print("  3. Enable motor control ('e' key)")
    print("  4. Boat will stay stationary until trash detected")
    print(f"  5. When trash detected: {detection_buffer_time}s buffer to place trash")
    print("  6. After buffer: Auto approaches and collects")
    print("  7. After collection: Returns to HOLD (stationary)")
    print("================================\n")
else:
    print("\nNormal AUTO mode - Mode switching is automatic")

avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 30

# Initialize Pixhawk connection variable
vehicle = None

# Initialize a flag to track if we've already forced motors off 
# (to avoid multiple redundant messages)
motors_forced_off = False

"""Attempt initial Pixhawk connection if enabled in config."""
if DRONEKIT_AVAILABLE:
    if pixhawk_enabled:
        print(f"\nConnecting to Pixhawk on {pixhawk_connection} at {pixhawk_baud} baud...")
        port = pixhawk_connection
        baud = pixhawk_baud
        # Check if port exists and has proper permissions
        if not os.path.exists(port):
            print(f"ERROR: Port {port} does not exist!")
            print("UART may not be enabled. Try running:")
            print("sudo raspi-config > Interface Options > Serial")
        else:
            # Try to fix permissions if needed
            if not os.access(port, os.R_OK | os.W_OK):
                print(f"WARNING: Permission issues on {port}")
                print("Attempting to fix permissions...")
                try:
                    import subprocess
                    subprocess.run(['sudo', 'chmod', '666', port])
                    print("Permissions updated.")
                except:
                    print(f"Could not update permissions. Try running: sudo chmod 666 {port}")
            # Try connection
            try:
                connection_success = connect_pixhawk(port, baud, timeout=pixhawk_timeout)
                if connection_success and vehicle is not None:
                    vehicle_status = get_vehicle_status()
                    print(f"SUCCESS! Connected on {port} at {baud} baud")
                    print(f"Vehicle status: {vehicle_status}")
                    # Save settings for reconnection if needed
                    successful_port = port
                    successful_baud = baud
                else:
                    print(f"Failed to connect to Pixhawk on {port} at {baud} baud.")
                    print("Check your physical connection and Pixhawk power.")
                    print("You can try reconnecting manually by pressing 'c' during operation.")
            except Exception as e:
                print(f"Connection error: {e}")
                print("You can try reconnecting manually by pressing 'c' during operation.")
    else:
        print("Pixhawk connection disabled via config (pixhawk.control_enabled = false)")
else:
    print("DroneKit is not available. Pixhawk connection disabled.")

try:
    collection_frame_counter = 0
    collection_frame_skip = max(1, 30 // collection_cam_fps) if collection_cam_enabled else 1
    
    while True:
        t_start = time.perf_counter()
        # Capture frame
        if source_type == 'usb':
            ret, frame = cap.read()
            if not ret:
                print('Unable to read from webcam. Check connection.')
                break
        else:
            try:
                frame = cap.capture_array()
                if frame is None:
                    print('Unable to read from Picamera. Check connection.')
                    break
                frame = np.ascontiguousarray(frame)
            except Exception as e:
                print(f'Error capturing from Picamera: {e}')
                break
        
        # Process collection camera (if enabled, at reduced FPS)
        if collection_cam_enabled and collection_cap is not None:
            collection_frame_counter += 1
            if collection_frame_counter >= collection_frame_skip:
                collection_frame_counter = 0
                
                ret_col, collection_frame = collection_cap.read()
                if ret_col:
                    # Run detection on collection camera
                    collection_results = model(collection_frame, verbose=False)
                    collection_detections = collection_results[0].boxes
                    
                    # Check if trash crossed collection line
                    check_collection_confirmation(collection_detections, collection_frame)
                    
                    # Display collection camera feed
                    cv2.imshow('Collection Camera', collection_frame)
        
        # Process and annotate frame
        frame, object_count = process_frame(frame)

        # Draw FPS and object count
        cv2.putText(frame, f'FPS: {avg_frame_rate:0.1f}', (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        cv2.putText(frame, f'Objects: {object_count}', (10,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        cv2.putText(frame, f'Confidence: {int(min_thresh*100)}%', (10,90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        
        # Display trash collection count
        cv2.putText(frame, f'Trash Collected: {total_trash_collected}', (10,270),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        # Display Pixhawk connection status if applicable
        if DRONEKIT_AVAILABLE:
            if vehicle is not None:
                connection_status = "Connected"
                color = (0,255,0)
            else:
                connection_status = "Not connected (Press 'c' to connect)"
                color = (0,0,255)
            cv2.putText(frame, f'Pixhawk: {connection_status}', (10,120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Display motor control status
        motor_status = "ACTIVE (LIVE)" if motor_control_enabled else "DISABLED (SAFE)"
        motor_color = (0,0,255) if motor_control_enabled else (0,255,0)
        cv2.putText(frame, f'Nav Thrust: {motor_status}', (10,150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, motor_color, 2)

        # Display mode switching status (always enabled)
        if test_nav_mode:
            nav_status = 'Buffering' if detection_buffering else 'Active'
            cv2.putText(frame, f'TEST_NAV MODE: {nav_status}', (10,180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,165,255), 2)
        else:
            cv2.putText(frame, f'Auto Mode Switch: ENABLED', (10,180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        # Display armed status
        if vehicle is not None:
            if vehicle.armed:
                cv2.putText(frame, f'ARMED', (10,210),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                try:
                    right_motor = vehicle.channels['1']
                    left_motor = vehicle.channels['3']
                    if (abs(right_motor - 1335) > 50 or abs(left_motor - 1500) > 50) and not motor_control_enabled:
                        warning = "WARNING: MOTORS ACTIVE!"
                        cv2.putText(frame, warning, (resW//2 - 150, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
                        motor_text = f"R:{right_motor} L:{left_motor}"
                        cv2.putText(frame, motor_text, (resW//2 - 80, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                except Exception:
                    pass
            else:
                cv2.putText(frame, f'DISARMED', (10,210),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.putText(frame, f'Mode: {vehicle.mode.name}', (10,240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

        frame_center = (resW // 2, resH // 2)
        cv2.line(frame, (frame_center[0], 0), (frame_center[0], resH), (0, 255, 255), 2)
        cv2.line(frame, (0, frame_center[1]), (resW, frame_center[1]), (0, 255, 255), 2)
        cv2.imshow('YOLO Detection', frame)
        if record:
            recorder.write(frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.waitKey(0)
        elif key == ord('p'):
            cv2.imwrite('capture.png', frame)
        elif key == ord('d'):
            print("\nRunning diagnostic port scan...")
            diagnostic_port_scan()
        elif key == ord('c'):
            if DRONEKIT_AVAILABLE:
                if vehicle is not None:
                    try:
                        test_mode = vehicle.mode.name
                        print("\nPixhawk is still connected. Current status:")
                        vehicle_status = get_vehicle_status()
                        print(f"Vehicle status: {vehicle_status}")
                    except Exception:
                        print("\nConnection lost. Attempting to reconnect...")
                        vehicle = None
                if vehicle is None:
                    port = pixhawk_connection
                    baud = pixhawk_baud
                    print(f"\nAttempting to reconnect to Pixhawk on {port} at {baud} baud...")
                    if not os.path.exists(port):
                        print(f"ERROR: Port {port} does not exist!")
                    else:
                        try:
                            if connect_pixhawk(port, baud, timeout=pixhawk_timeout):
                                vehicle_status = get_vehicle_status()
                                print(f"Reconnected! Vehicle status: {vehicle_status}")
                            else:
                                print(f"Failed to reconnect to Pixhawk on {port} at {baud} baud.")
                                print("Check your physical connection and Pixhawk power.")
                        except Exception as e:
                            print(f"Reconnection error: {e}")
            else:
                print("DroneKit is not available. Cannot connect to Pixhawk.")
        elif key == ord('h'):
            show_command_help()
        elif key == ord('m'):
            if vehicle:
                # Filtered to common ArduRover / general modes. Removed Copter-only modes like STABILIZE, ALT_HOLD, LOITER, LAND.
                available_modes = ['MANUAL', 'HOLD', 'GUIDED', 'AUTO', 'RTL']
                print("\nAvailable flight modes:", ", ".join(available_modes))
                print("Current mode:", vehicle.mode.name)
                print("NOTE: Mode switching is now automatic based on detections")
                print("Current mode:", vehicle.mode.name)
                print("Enter first letter of mode or full mode name (e.g., 's' or 'STABILIZE')")
                cv2.putText(frame, "Enter mode in terminal", (resW//3, resH//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
                cv2.imshow('YOLO Detection', frame)
                cv2.waitKey(1)
                mode_input = input("New mode: ").strip().upper()
                if len(mode_input) == 1:
                    for mode in available_modes:
                        if mode.startswith(mode_input):
                            mode_input = mode
                            break
                if mode_input in available_modes:
                    set_flight_mode(mode_input)
                else:
                    print(f"Invalid mode: {mode_input}")
            else:
                print("Pixhawk not connected! Connect first.")
        elif key == ord('a'):
            if vehicle:
                if arm_disarm(True):
                    print("Setting all motors to neutral position after arming...")
                    set_neutral_motors()
            else:
                print("Pixhawk not connected! Connect first.")
        elif key == ord('f'):
            if vehicle:
                arm_disarm(False)
            else:
                print("Pixhawk not connected! Connect first.")
        elif key == ord('l'):
            # Steer LEFT test
            if not vehicle:
                print("Pixhawk not connected! Connect first.")
            else:
                if not vehicle.armed:
                    print("Vehicle not armed. Arm ('a') before motor test.")
                else:
                    print("\nSTEER LEFT TEST -- Safety: Ensure vessel secure")
                    print("Setting Ch1=935 (left), Ch3=1600 (fwd) for 5 seconds...")
                    try:
                        vehicle.channels.overrides['1'] = 935   # Steer left
                        vehicle.channels.overrides['3'] = 1600  # Forward
                        time.sleep(5.0)
                    finally:
                        vehicle.channels.overrides['1'] = 1335  # Steering neutral
                        vehicle.channels.overrides['3'] = 1500  # Throttle neutral
                    print("Steer left test complete.")
        elif key == ord('k'):
            # Reverse throttle test
            if not vehicle:
                print("Pixhawk not connected! Connect first.")
            else:
                if not vehicle.armed:
                    print("Vehicle not armed. Arm ('a') before motor test.")
                else:
                    print("\nREVERSE THROTTLE TEST -- Safety: Ensure vessel secure")
                    print("Setting Ch3=1100 (reverse), Ch1=1335 (straight) for 5 seconds...")
                    try:
                        vehicle.channels.overrides['1'] = 1335  # Steering neutral
                        vehicle.channels.overrides['3'] = 1100  # Reverse
                        time.sleep(5.0)
                    finally:
                        vehicle.channels.overrides['3'] = 1500  # Throttle neutral
                    print("Reverse throttle test complete.")
        elif key == ord('r'):
            # Steer RIGHT test
            if not vehicle:
                print("Pixhawk not connected! Connect first.")
            else:
                if not vehicle.armed:
                    print("Vehicle not armed. Arm ('a') before motor test.")
                else:
                    print("\nSTEER RIGHT TEST -- Safety: Ensure vessel secure")
                    print("Setting Ch1=1735 (right), Ch3=1600 (fwd) for 5 seconds...")
                    try:
                        vehicle.channels.overrides['1'] = 1735  # Steer right
                        vehicle.channels.overrides['3'] = 1600  # Forward
                        time.sleep(5.0)
                    finally:
                        vehicle.channels.overrides['1'] = 1335  # Steering neutral
                        vehicle.channels.overrides['3'] = 1500  # Throttle neutral
                    print("Steer right test complete.")
        elif key == ord('t'):
            # Forward throttle test
            if not vehicle:
                print("Pixhawk not connected! Connect first.")
            else:
                if not vehicle.armed:
                    print("Vehicle not armed. Arm ('a') before motor test.")
                else:
                    print("\nFORWARD THROTTLE TEST -- Safety: Ensure vessel secure")
                    print("Setting Ch3=1894 (forward), Ch1=1335 (straight) for 5 seconds...")
                    try:
                        vehicle.channels.overrides['1'] = 1335  # Steering neutral
                        vehicle.channels.overrides['3'] = 1894  # Forward
                        time.sleep(5.0)
                    finally:
                        vehicle.channels.overrides['3'] = 1500  # Throttle neutral
                    print("Forward throttle test complete.")
        elif key == ord('e'):
            motor_control_enabled = not motor_control_enabled
            status = "ACTIVE" if motor_control_enabled else "INACTIVE"
            print(f"\n[DIVERSION] Auto navigation thrust is now {status}")
            temp_frame = frame.copy()
            message = f"Auto Nav: {status}"
            textsize = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)[0]
            cv2.putText(temp_frame, message, (resW//2 - textsize[0]//2, resH//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0,0,255) if motor_control_enabled else (0,255,0), 3)
            if not motor_control_enabled and vehicle and vehicle.armed:
                print("Disabling auto thrust and setting motors to neutral")
                motor_active = False
                set_neutral_motors()
        elif key == ord('b'):
            if vehicle:
                print("\nSending reboot command to Pixhawk...")
                # MAV_CMD_PREFLIGHT_REBOOT_SHUTDOWN = 246
                # param1 = 1 (Reboot autopilot)
                try:
                    msg = vehicle.message_factory.command_long_encode(
                        0, 0,    # target_system, target_component
                        246,     # command
                        0,       # confirmation
                        1,       # param1: 1=Reboot autopilot
                        0, 0, 0, 0, 0, 0 # param2-7
                    )
                    vehicle.send_mavlink(msg)
                    print("Reboot command sent. Connection will be lost.")
                    vehicle = None
                except Exception as e:
                    print(f"Error sending reboot command: {e}")
            else:
                print("Pixhawk not connected! Connect first.")
        elif key == ord('z'):
            if vehicle:
                if set_flight_mode('RTL'):
                    print("Vehicle returning to launch point")
                else:
                    print("Failed to enter RTL mode")
            else:
                print("Pixhawk not connected! Connect first.")
        t_stop = time.perf_counter()
        frame_rate = 1/(t_stop - t_start)
        frame_rate_buffer.append(frame_rate)
        if len(frame_rate_buffer) > fps_avg_len:
            frame_rate_buffer.pop(0)
        avg_frame_rate = np.mean(frame_rate_buffer)
except KeyboardInterrupt:
    print("Interrupted by user")
except Exception as e:
    print(f"Unhandled exception in main loop: {e}")
finally:
    # ---------------------------
    # Cleanup (robust)
    # ---------------------------
    print(f'Average FPS: {avg_frame_rate:.1f}')
    print(f'Total Trash Collected: {total_trash_collected}')
    
    try:
        if source_type == 'usb':
            cap.release()
        else:
            cap.stop()
    except Exception:
        pass
    
    # Release collection camera
    if collection_cap is not None:
        try:
            collection_cap.release()
            print("Collection camera released.")
        except Exception as e:
            print(f"Error releasing collection camera: {e}")
    
    if record:
        try:
            recorder.release()
        except Exception:
            pass
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass
    if vehicle is not None:
        try:
            print("Turning off all motor outputs...")
            control_motors(right_motor_value=1335, left_motor_value=1500)
            clear_all_motor_overrides()
            set_neutral_motors()
            time.sleep(0.3)
            vehicle.channels.overrides = {}
            print("Motor shutdown sequence complete")
        except Exception as e:
            print(f"Error during motor shutdown: {e}")
        try:
            vehicle.close()
            print("Pixhawk connection closed.")
        except Exception as e:
            print(f"Error closing vehicle: {e}")

# ---------------------------
# Cleanup
# ---------------------------
