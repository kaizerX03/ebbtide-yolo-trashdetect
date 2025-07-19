import os
import sys
import time
import yaml

import cv2
import numpy as np
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

def load_config():
    """Load YAML configuration file."""
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'detection_config.yaml')
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f'ERROR: Could not load configuration file: {e}')
        sys.exit(1)

config = load_config()
model_path = config['model']['path']
img_source = config['camera']['source']
min_thresh = float(config['model']['confidence_threshold'])
resW, resH = map(int, config['camera']['resolution'].split('x'))
record = config['recording']['enabled']
distance_estimation_config = config.get('distance_estimation', {})
focal_length_px = distance_estimation_config.get('focal_length_px')
known_object_height_mm = distance_estimation_config.get('known_object_height_mm')

if not os.path.exists(model_path):
    print('ERROR: Model path is invalid or model was not found.')
    sys.exit(0)

# ---------------------------
# Model and Camera Initialization
# ---------------------------

model = YOLO(model_path, task='detect')
labels = model.names

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
# Pixhawk Integration
# ---------------------------
vehicle = None

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
        vehicle = connect(connection_string, baud=baud_rate, wait_ready=False, timeout=timeout)
        print("Initial connection established...")
        
        # Now wait for parameters to download
        print("Waiting for vehicle to initialize...")
        vehicle.wait_ready(True, timeout=timeout)
        
        print(f"Connected to vehicle!")
        print(f" > System status: {vehicle.system_status.state}")
        print(f" > Mode: {vehicle.mode.name}")
        print(f" > GPS: {vehicle.gps_0.fix_type}")
        print(f" > Battery: {vehicle.battery.voltage}V")
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
                print("WARNING: Vehicle is not armable! Check GPS, EKF, and compass.")
                print(f"Current status: {vehicle.system_status.state}")
                return False
                
            print("Arming motors...")
            vehicle.armed = True
            
            # Wait for arming to take effect
            for _ in range(10):  # Try for a few seconds
                if vehicle.armed:
                    print("Vehicle armed!")
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

def show_command_help():
    """Display available commands for the Pixhawk."""
    print("\n--- Pixhawk Commands ---")
    print("  'm': Change flight mode")
    print("  'a': Arm motors")
    print("  'f': Disarm motors")
    print("  'h': Show this help menu")
    print("  't': Takeoff (to 2m altitude)")
    print("  'l': Land")
    print("  'z': Return to Launch (RTL)")
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

def process_frame(frame):
    """Run detection and annotate frame."""
    results = model(frame, verbose=False)
    detections = results[0].boxes
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

avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 30

# Initialize Pixhawk connection variable
vehicle = None

# Automatically connect to Pixhawk on startup
if DRONEKIT_AVAILABLE:
    print("\nConnecting to Pixhawk on /dev/ttyAMA0 at 57600 baud...")
    
    # Set known good connection parameters
    port = '/dev/ttyAMA0'
    baud = 57600
    
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
            connection_success = connect_pixhawk(port, baud)
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
    print("DroneKit is not available. Pixhawk connection disabled.")

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
    # Process and annotate frame
    frame, object_count = process_frame(frame)
    
    # Draw FPS and object count
    cv2.putText(frame, f'FPS: {avg_frame_rate:0.1f}', (10,30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    cv2.putText(frame, f'Objects: {object_count}', (10,60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    
    # Display Pixhawk connection status if applicable
    if DRONEKIT_AVAILABLE:
        if vehicle is not None:
            connection_status = "Connected"
            color = (0,255,0)  # Green
        else:
            connection_status = "Not connected (Press 'c' to connect)"
            color = (0,0,255)  # Red
        cv2.putText(frame, f'Pixhawk: {connection_status}', (10,90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
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
        cv2.waitKey(0)  # Pause
    elif key == ord('p'):
        cv2.imwrite('capture.png', frame)  # Save image
    elif key == ord('d'):
        # Run port diagnostic
        print("\nRunning diagnostic port scan...")
        diagnostic_port_scan()
    elif key == ord('c'):
        # Try to reconnect to Pixhawk if disconnected
        if DRONEKIT_AVAILABLE:
            if vehicle is not None:
                try:
                    # Test if the connection is still alive
                    test_mode = vehicle.mode.name
                    print("\nPixhawk is still connected. Current status:")
                    vehicle_status = get_vehicle_status()
                    print(f"Vehicle status: {vehicle_status}")
                except Exception:
                    print("\nConnection lost. Attempting to reconnect...")
                    vehicle = None
            
            if vehicle is None:
                # Use known port and baud rate for connection
                port = '/dev/ttyAMA0'
                baud = 57600
                
                print(f"\nAttempting to reconnect to Pixhawk on {port} at {baud} baud...")
                
                # Check if port exists and has proper permissions
                if not os.path.exists(port):
                    print(f"ERROR: Port {port} does not exist!")
                else:
                    # Attempt to connect
                    try:
                        if connect_pixhawk(port, baud):
                            vehicle_status = get_vehicle_status()
                            print(f"Reconnected! Vehicle status: {vehicle_status}")
                        else:
                            print(f"Failed to reconnect to Pixhawk on {port} at {baud} baud.")
                            print("Check your physical connection and Pixhawk power.")
                    except Exception as e:
                        print(f"Reconnection error: {e}")
        else:
            print("DroneKit is not available. Cannot connect to Pixhawk.")
    # Pixhawk command keys
    elif key == ord('h'):
        # Show help for Pixhawk commands
        show_command_help()
    elif key == ord('m'):
        # Change flight mode
        if vehicle:
            available_modes = ['STABILIZE', 'ALT_HOLD', 'LOITER', 'GUIDED', 'AUTO', 'RTL', 'LAND']
            print("\nAvailable flight modes:", ", ".join(available_modes))
            print("Current mode:", vehicle.mode.name)
            print("Enter first letter of mode or full mode name (e.g., 's' or 'STABILIZE')")
            cv2.putText(frame, "Enter mode in terminal", (resW//3, resH//2), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
            cv2.imshow('YOLO Detection', frame)
            cv2.waitKey(1)
            
            mode_input = input("New mode: ").strip().upper()
            
            # Handle single letter input
            if len(mode_input) == 1:
                for mode in available_modes:
                    if mode.startswith(mode_input):
                        mode_input = mode
                        break
            
            # Verify mode is valid
            if mode_input in available_modes:
                set_flight_mode(mode_input)
            else:
                print(f"Invalid mode: {mode_input}")
        else:
            print("Pixhawk not connected! Connect first.")
    elif key == ord('a'):
        # Arm motors
        if vehicle:
            arm_disarm(True)
        else:
            print("Pixhawk not connected! Connect first.")
    elif key == ord('f'):
        # Disarm motors
        if vehicle:
            arm_disarm(False)
        else:
            print("Pixhawk not connected! Connect first.")
    elif key == ord('t'):
        # Takeoff to 2 meters
        if vehicle:
            if set_flight_mode('GUIDED'):  # Must be in GUIDED mode to takeoff
                takeoff(2.0)  # 2 meters altitude
            else:
                print("Failed to enter GUIDED mode for takeoff")
        else:
            print("Pixhawk not connected! Connect first.")
    elif key == ord('l'):
        # Land
        if vehicle:
            set_flight_mode('LAND')
        else:
            print("Pixhawk not connected! Connect first.")
    elif key == ord('z'):
        # Return to launch
        if vehicle:
            set_flight_mode('RTL')
        else:
            print("Pixhawk not connected! Connect first.")
    t_stop = time.perf_counter()
    frame_rate = 1/(t_stop - t_start)
    frame_rate_buffer.append(frame_rate)
    if len(frame_rate_buffer) > fps_avg_len:
        frame_rate_buffer.pop(0)
    avg_frame_rate = np.mean(frame_rate_buffer)

# ---------------------------
# Cleanup
# ---------------------------

print(f'Average FPS: {avg_frame_rate:.1f}')
if source_type == 'usb':
    cap.release()
else:
    cap.stop()
if record:
    recorder.release()
cv2.destroyAllWindows()

# Close Pixhawk connection
if vehicle is not None:
    vehicle.close()
    print("Pixhawk connection closed.")
