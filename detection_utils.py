import cv2
import numpy as np

def process_frame(frame, model, labels, min_thresh, bbox_colors, focal_length_px=None, known_object_height_mm=None, resW=None, resH=None):
    """Run detection and annotate frame."""
    results = model(frame, verbose=False)
    detections = results[0].boxes
    object_count = 0
    most_centered_idx = None
    min_center_dist = None
    frame_center = (resW // 2, resH // 2) if resW and resH else (0, 0)
    centers = []
    for i, detection in enumerate(detections):
        xyxy = detection.xyxy.cpu().numpy().squeeze()
        xmin, ymin, xmax, ymax = xyxy.astype(int)
        classidx = int(detection.cls.item())
        classname = labels[classidx]
        conf = detection.conf.item()
        if conf > min_thresh:
            color = bbox_colors[classidx % len(bbox_colors)]
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            label = f'{classname}: {int(conf*100)}%'
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
            center_x = (xmin + xmax) // 2
            center_y = (ymin + ymax) // 2
            centers.append((center_x, center_y))
            dist = np.hypot(center_x - frame_center[0], center_y - frame_center[1])
            if min_center_dist is None or dist < min_center_dist:
                min_center_dist = dist
                most_centered_idx = i
    if most_centered_idx is not None and centers:
        cx, cy = centers[most_centered_idx]
        cv2.drawMarker(frame, (cx, cy), (0,255,0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
        cv2.putText(frame, 'MOST CENTERED', (cx-60, cy-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    return frame, object_count
