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

# Load configuration from YAML file
config_path = os.path.join(os.path.dirname(__file__), 'config', 'detection_config.yaml')
try:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
except Exception as e:
    print(f'ERROR: Could not load configuration file: {e}')
    sys.exit(1)

# Parse configuration
model_path = config['model']['path']
img_source = config['camera']['source']
min_thresh = float(config['model']['confidence_threshold'])
resW, resH = map(int, config['camera']['resolution'].split('x'))
record = config['recording']['enabled']

# Check if model file exists
if not os.path.exists(model_path):
    print('ERROR: Model path is invalid or model was not found.')
    sys.exit(0)

# Load the model and get labelmap
model = YOLO(model_path, task='detect')
labels = model.names

# Initialize camera based on source type
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
    
    # Create a more optimized configuration for Pi Camera
    config = cap.create_video_configuration(
        main={"format": 'RGB888', "size": (resW, resH)},
        lores={"format": 'YUV420', "size": (resW // 2, resH // 2)},
        display="lores",
        buffer_count=4  # Increase buffer for better performance
    )
    
    # Configure camera with tuned parameters
    cap.configure(config)
    
    # Set camera controls for better performance
    cap.set_controls({
        "FrameDurationLimits": (33333, 33333),  # Target 30fps
        "AnalogueGain": 1.0,
        "ExposureTime": 20000,
        "AeEnable": True,  # Auto exposure
        "AwbEnable": True,  # Auto white balance
    })
    
    # Start the camera
    cap.start()
else:
    print('Invalid source. Use "usb0" for webcam or "picamera0" for Pi Camera.')
    sys.exit(0)

# Set up recording if requested
if record:
    record_name = 'demo1.avi'
    record_fps = 30
    recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (resW, resH))

# Set bounding box colors (Tableau 10 color scheme)
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), 
              (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

# Initialize FPS calculation
avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 30

# Main detection loop
while True:
    t_start = time.perf_counter()

    # Capture frame
    if source_type == 'usb':
        ret, frame = cap.read()
        if not ret:
            print('Unable to read from webcam. Check connection.')
            break
    else:  # picamera
        try:
            # Capture frame directly in RGB format (more efficient)
            frame = cap.capture_array()
            if frame is None:
                print('Unable to read from Picamera. Check connection.')
                break
            
            # No need for color conversion since we configured RGB888 format
            frame = np.ascontiguousarray(frame)  # Ensure memory is contiguous for better performance
        except Exception as e:
            print(f'Error capturing from Picamera: {e}')
            break

    # Run inference
    results = model(frame, verbose=False)
    detections = results[0].boxes
    object_count = 0

    # Process detections
    for detection in detections:
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
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), 
                         (xmin+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED)
            cv2.putText(frame, label, (xmin, label_ymin-7), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            object_count += 1

    # Draw FPS and object count
    cv2.putText(frame, f'FPS: {avg_frame_rate:0.1f}', (10,30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    cv2.putText(frame, f'Objects: {object_count}', (10,60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    # Display frame and record if enabled
    cv2.imshow('YOLO Detection', frame)
    if record:
        recorder.write(frame)

    # Handle key presses
    key = cv2.waitKey(1)
    if key == ord('q'):  # Quit
        break
    elif key == ord('s'):  # Pause
        cv2.waitKey(0)
    elif key == ord('p'):  # Save frame
        cv2.imwrite('capture.png', frame)

    # Calculate FPS
    t_stop = time.perf_counter()
    frame_rate = 1/(t_stop - t_start)
    frame_rate_buffer.append(frame_rate)
    if len(frame_rate_buffer) > fps_avg_len:
        frame_rate_buffer.pop(0)
    avg_frame_rate = np.mean(frame_rate_buffer)

# Cleanup
print(f'Average FPS: {avg_frame_rate:.1f}')
if source_type == 'usb':
    cap.release()
else:
    cap.stop()
if record:
    recorder.release()
cv2.destroyAllWindows()

os.system('python yolo_detect.py')
