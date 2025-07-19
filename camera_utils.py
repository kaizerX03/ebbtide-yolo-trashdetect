import cv2
import time

# Optional Picamera2 import
PICAMERA_AVAILABLE = False
try:
    from picamera2 import Picamera2
    PICAMERA_AVAILABLE = True
except ImportError:
    pass

def initialize_camera(img_source, resW, resH):
    """Initialize camera based on source type."""
    if 'usb' in img_source:
        usb_idx = int(img_source[3:])
        cap = cv2.VideoCapture(usb_idx)
        cap.set(3, resW)
        cap.set(4, resH)
        return cap, 'usb'
    elif 'picamera' in img_source:
        if not PICAMERA_AVAILABLE:
            raise ImportError('Picamera2 module not available but picamera source specified.')
        cap = Picamera2()
        config_cam = cap.create_video_configuration(
            main={"format": 'RGB888', "size": (resW, resH)}
        )
        cap.configure(config_cam)
        cap.start()
        time.sleep(1.0)
        return cap, 'picamera'
    else:
        raise ValueError('Invalid camera source. Use "usb0" or "picamera0".')
