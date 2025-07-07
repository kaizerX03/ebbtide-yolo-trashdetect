import cv2
import numpy as np
from ultralytics import YOLO
import os
import time

# Make Picamera2 import optional
PICAMERA_AVAILABLE = False
try:
    from picamera2 import Picamera2
    PICAMERA_AVAILABLE = True
except ImportError:
    print("Warning: Picamera2 module not available. USB camera will be used if available.")

# --- USER CONFIGURATION ---
# INSTRUCTIONS:
# 1. Set the known distance and height of your calibration object below.
# 2. Run the script. A live camera feed will appear.
# 3. Position your object at the known distance.
# 4. Ensure the object is detected correctly (a box appears around it).
# 5. Press 'c' to perform the calibration calculation.
# 6. Press 'q' to quit.

# -- Values to update --
KNOWN_DISTANCE_MM = 600  # Distance from camera to object in millimeters
KNOWN_HEIGHT_MM = 100    # Actual height of the object in millimeters
MODEL_PATH = '/home/pi/yolo_env/ebb_ncnn_model' # Path to your YOLO model
TARGET_CLASS_ID = 2      # The class ID of the object you are using for calibration
CAMERA_SOURCE = "picamera0" # "picamera0" or "usb0"
RESOLUTION_W = 640
RESOLUTION_H = 480
# --------------------------

print("Live Camera Calibration Script")
print("------------------------------")
print(f"Instructions: Position the object (class ID: {TARGET_CLASS_ID}) at {KNOWN_DISTANCE_MM}mm from the camera.")
print("Press 'c' to calculate focal length when the object is detected.")
print("Press 'q' to quit.")

# --- Verification ---
if not os.path.exists(MODEL_PATH):
    print(f"ERROR: YOLO model not found at '{MODEL_PATH}'")
    exit()

# --- Initialize Camera ---
if 'picamera' in CAMERA_SOURCE and PICAMERA_AVAILABLE:
    cap = Picamera2()
    config = cap.create_video_configuration(main={"format": 'RGB888', "size": (RESOLUTION_W, RESOLUTION_H)})
    cap.configure(config)
    cap.start()
    time.sleep(1.0) # Allow camera to warm up
    print("Picamera initialized.")
else:
    if 'usb' in CAMERA_SOURCE:
        usb_idx = int(CAMERA_SOURCE.replace('usb', ''))
        cap = cv2.VideoCapture(usb_idx)
        if not cap.isOpened():
            print(f"ERROR: Could not open USB camera at index {usb_idx}.")
            exit()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION_H)
        print("USB camera initialized.")
    else:
        print("ERROR: No valid camera source found. Check CAMERA_SOURCE setting or Picamera2 installation.")
        exit()

# --- Calibration Logic ---

# Load the YOLO model
print(f"Loading model: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

# Main loop for live detection
while True:
    # Capture frame
    if 'picamera' in CAMERA_SOURCE and PICAMERA_AVAILABLE:
        frame = cap.capture_array()
    else:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to grab frame from camera.")
            break

    # Run detection on the frame
    results = model(frame, verbose=False)
    detections = results[0].boxes

    # Find the target object for this frame
    target_detected_this_frame = False
    
    # Draw boxes for all detections
    for detection in detections:
        xyxy = detection.xyxy.cpu().numpy().squeeze().astype(int)
        xmin, ymin, xmax, ymax = xyxy
        cls_id = int(detection.cls.item())
        class_name = model.names[cls_id]
        conf = detection.conf.item()
        
        color = (0, 0, 255) # Default to red
        if cls_id == TARGET_CLASS_ID:
            color = (0, 255, 0) # Green for the target object
            target_detected_this_frame = True

        label = f"{class_name} (ID: {cls_id})"
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Display instructions on the frame
    cv2.putText(frame, "Press 'c' to Calibrate, 'q' to Quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Show the frame
    cv2.imshow("Live Calibration", frame)
    key = cv2.waitKey(1) & 0xFF

    # Handle key presses
    if key == ord('q'):
        print("Quitting...")
        break
    
    if key == ord('c'):
        if not target_detected_this_frame:
            print("Calibration failed: Target object was not detected in the frame. Please try again.")
            continue

        print("\nCalculating focal length...")
        
        # Find the target object again to get its details for calculation
        for detection in detections:
            if int(detection.cls.item()) == TARGET_CLASS_ID:
                xyxy = detection.xyxy.cpu().numpy().squeeze().astype(int)
                object_pixel_height = xyxy[3] - xyxy[1]
                
                # Formula: Focal Length (px) = (Pixel Height * Known Distance) / Known Height
                focal_length_px = (object_pixel_height * KNOWN_DISTANCE_MM) / KNOWN_HEIGHT_MM

                # --- Output Results ---
                print("\n--- Calibration Successful ---")
                print(f"Object detected with pixel height: {object_pixel_height}px")
                print(f"Calculated Focal Length: {focal_length_px:.2f} pixels")
                print("------------------------------")
                print("\nAction Required:")
                print("1. Open your 'config/detection_config.yaml' file.")
                print("2. Add or update the 'distance_estimation' section with the calculated focal length:")
                print("\ndistance_estimation:")
                print(f"  focal_length_px: {focal_length_px:.2f}")
                print(f"  known_object_height_mm: {KNOWN_HEIGHT_MM}")

                # Save a debug image
                output_dir = 'calibration_output'
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                output_path = os.path.join(output_dir, 'live_calibration_capture.jpg')
                cv2.imwrite(output_path, frame)
                print(f"\nSaved calibration capture to: {output_path}")
                
                # Exit after successful calibration
                break
        break # Exit the main while loop

# --- Cleanup ---
if 'picamera' in CAMERA_SOURCE and PICAMERA_AVAILABLE:
    cap.stop()
else:
    cap.release()
cv2.destroyAllWindows()
