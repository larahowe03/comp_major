import cv2
import pickle
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO
from collections import deque
import time
import copy
from ultralytics.engine.results import Boxes
from warp_board import process_chess_image
from split_dataset_for_YOLO_contour import preprocess_image

# Model paths
contour_model_path = "runs_chess/final_model_warp_contour/weights/best.pt"
colour_model_path = "runs_chess/final_model_warp_colour/weights/best.pt"

# Load models
device = 0 if torch.cuda.is_available() else 'cpu'
contour_model = YOLO(contour_model_path)
colour_model = YOLO(colour_model_path)

print(f"Using device: {device}")

def check_colour(bbox_img):
    hsv = cv2.cvtColor(bbox_img, cv2.COLOR_BGR2HSV)
    
    # Define color ranges
    lower_cream = np.array([90, 50, 50])
    upper_cream = np.array([130, 255, 255])
    lower_dark = np.array([0, 0, 0])
    upper_dark = np.array([180, 255, 60])
    
    # Create masks
    mask_cream = cv2.inRange(hsv, lower_cream, upper_cream)
    mask_dark = cv2.inRange(hsv, lower_dark, upper_dark)
    
    # Count non-zero pixels in each mask
    cream_pixels = cv2.countNonZero(mask_cream)
    dark_pixels = cv2.countNonZero(mask_dark)
    
    # Determine which color dominates
    if cream_pixels > dark_pixels:
        return 'white'
    else:
        return 'black'

def contour_model_prediction(img):
    result = contour_model.predict(
        source=img,
        conf=0.2,
        imgsz=448,
        device=device,
        verbose=False
    )
    return result[0]


def colour_model_prediction(img):
    result = colour_model.predict(
        source=img,
        conf=0.2,
        imgsz=448,
        device=device,
        verbose=False
    )
    return result[0]

def undistort(img, K, d):
    return cv2.undistort(img, K, d, None, K)


# Initialize camera and calibration
def initialize_camera(phone_ip="10.19.206.177", port="4747"):
    """Initialize camera connection."""
    urls = [
        f"http://{phone_ip}:{port}/video",
        f"http://{phone_ip}:{port}/mjpegfeed",
    ]
    
    for url in urls:
        print(f"Trying: {url}")
        cap = cv2.VideoCapture(url)
        if cap.isOpened():
            print(f"Connected successfully to {url}")
            return cap
        cap.release()
    
    print("Could not connect to camera")
    return None


def load_calibration(filepath="calibration_coefficients.pkl"):
    """Load camera calibration data."""
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data['camera_matrix'], data['distortion_coeffs']


# Global variables for camera and calibration
camera = None
K_matrix = None
dist_coeffs = None
# stabilizer = None


def init_detection_system():
    """Initialize the detection system (call once at startup)."""
    global camera, K_matrix, dist_coeffs, stabilizer
    
    camera = initialize_camera()
    if camera is None:
        return False
    
    K_matrix, dist_coeffs = load_calibration()
    # stabilizer = PredictionStabilizer(window_seconds=1.0, max_distance=50)
    
    return True

def get_current_frame():
    """Get and process current frame from camera."""
    global camera, K_matrix, dist_coeffs
    
    if camera is None:
        return None, None, None
    
    ret, frame = camera.read()
    if not ret:
        return None, None, None
    
    # Undistort
    undistorted = undistort(frame, K_matrix, dist_coeffs)
    
    # Warp board
    warp_margined, warp_unmargined, contoured_img, pts_src = process_chess_image(undistorted)
    
    if warp_margined is None:
        return undistorted, undistorted, None, None
    
    return warp_margined, warp_unmargined, contoured_img, pts_src

def detect_pieces(warped):
    """Detect chess pieces on warped board."""
    
    if warped is None:
        return None
    
    # Prepare images
    contoured_warp = preprocess_image(warped)
    coloured_warp = warped
    
    contour_predictions = contour_model_prediction(contoured_warp)
    colour_predictions = colour_model_prediction(coloured_warp)
    
    # Apply stabilization
    # stabilized_contour = stabilizer.update(contour_predictions)
    # stabilized_colour = stabilizer.update(colour_predictions)
    
    contour_annotated = contour_predictions.plot()
    colour_annotated = colour_predictions.plot()

    # Extract all boxes from contour model
    contour_boxes = []
    contour_classes = []
    piece_colours = []
    for box in contour_predictions.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        class_id = int(box.cls[0].cpu().numpy())
        class_name = contour_predictions.names[class_id]
        contour_boxes.append({
            "x1": x1,
            "x2": x2,
            "y1": y1,
            "y2": y2})
        contour_classes.append(class_name)
        
        boxed_region = warped[y1:y2, x1:x2]
        piece_colours.append(check_colour(boxed_region))
    
    # Extract all boxes from colour model
    colour_boxes = []
    colour_classes = []
    for box in colour_predictions.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        class_id = int(box.cls[0].cpu().numpy())
        class_name = colour_predictions.names[class_id]
        colour_boxes.append({
            "x1": x1,
            "x2": x2,
            "y1": y1,
            "y2": y2})
        colour_classes.append(class_name)

    return contour_boxes, colour_boxes, contour_classes, colour_classes, contour_annotated, colour_annotated, piece_colours


def cleanup_camera():
    """Clean up camera resources."""
    global camera
    if camera is not None:
        camera.release()
        cv2.destroyAllWindows()