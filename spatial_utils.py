"""
Utilities for accessing and displaying spatial data from Pixhawk.

This module provides functions for retrieving GPS coordinates, 
heading, and other navigation data from a connected Pixhawk,
as well as displaying this information on video frames.
"""

import math
import time
import cv2
import numpy as np

def get_spatial_data(vehicle):
    """
    Get spatial data from Pixhawk for navigation purposes.
    
    Args:
        vehicle: DroneKit vehicle object
        
    Returns:
        Dictionary containing GPS, heading, and attitude data,
        or None if vehicle is not connected or error occurs
    """
    if vehicle is None:
        return None
        
    try:
        data = {
            # GPS Position
            "lat": vehicle.location.global_frame.lat if vehicle.location.global_frame else None,
            "lon": vehicle.location.global_frame.lon if vehicle.location.global_frame else None,
            "alt": vehicle.location.global_relative_frame.alt if vehicle.location.global_relative_frame else None,
            
            # Heading
            "heading": vehicle.heading,  # 0-360 degrees (0=North)
            
            # Attitude
            "roll": round(math.degrees(vehicle.attitude.roll), 1),
            "pitch": round(math.degrees(vehicle.attitude.pitch), 1),
            "yaw": round(math.degrees(vehicle.attitude.yaw), 1),
            
            # Velocity
            "groundspeed": vehicle.groundspeed,  # m/s
            
            # System
            "armed": vehicle.armed,
            "mode": vehicle.mode.name,
            
            # GPS info
            "gps_fix": vehicle.gps_0.fix_type,  # 0-1=No fix, 2=2D fix, 3=3D fix
            "satellites": vehicle.gps_0.satellites_visible,
            
            # Timestamp
            "timestamp": time.time()
        }
        return data
    except Exception as e:
        print(f"Error getting spatial data: {e}")
        return None

def display_spatial_data(frame, data):
    """
    Display spatial data overlay on video feed.
    
    Args:
        frame: OpenCV image frame
        data: Dictionary containing spatial data from get_spatial_data()
    """
    if data is None:
        return
    
    # Create semi-transparent overlay for data
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (w-300, 0), (w, 180), (0, 0, 0), -1)
    
    # GPS fix status colors
    if data['gps_fix'] >= 3:  # 3D Fix
        gps_color = (0, 255, 0)  # Green
        gps_text = "3D FIX"
    elif data['gps_fix'] == 2:  # 2D Fix
        gps_color = (0, 255, 255)  # Yellow
        gps_text = "2D FIX"
    else:  # No fix
        gps_color = (0, 0, 255)  # Red
        gps_text = "NO FIX"
    
    # Display spatial data
    cv2.putText(overlay, f"GPS: {data['lat']:.6f}, {data['lon']:.6f}", 
                (w-290, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(overlay, f"ALT: {data['alt']:.1f}m", 
                (w-290, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(overlay, f"HDG: {data['heading']}° SPEED: {data['groundspeed']:.1f}m/s", 
                (w-290, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(overlay, f"ROLL: {data['roll']}°  PITCH: {data['pitch']}°", 
                (w-290, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(overlay, f"GPS: {gps_text} ({data['satellites']} sats)", 
                (w-290, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, gps_color, 1)
    
    # Add a compass indicator
    compass_center = (w-150, 150)
    compass_radius = 40
    # Handle the case where heading is None
    heading = data['heading'] if data['heading'] is not None else 0
    heading_rad = math.radians(heading)
    needle_end = (
        int(compass_center[0] + compass_radius * math.sin(heading_rad)),
        int(compass_center[1] - compass_radius * math.cos(heading_rad))
    )
    
    # Draw compass circle
    cv2.circle(overlay, compass_center, compass_radius, (255, 255, 255), 1)
    
    # Draw cardinal directions
    cv2.putText(overlay, "N", (compass_center[0]-5, compass_center[1]-compass_radius-5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(overlay, "E", (compass_center[0]+compass_radius+5, compass_center[1]+5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(overlay, "S", (compass_center[0]-5, compass_center[1]+compass_radius+15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(overlay, "W", (compass_center[0]-compass_radius-15, compass_center[1]+5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Draw heading needle
    cv2.line(overlay, compass_center, needle_end, (0, 0, 255), 2)
    
    # Apply the overlay with transparency
    alpha = 0.7
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    return frame
