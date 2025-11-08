import cv2
import numpy as np
import torch
from ultralytics import YOLO
from src.training.split_dataset_for_YOLO_contour import preprocess_image

# ------------------------------------------------------------------------
# DETECTIONS WITH MODELS FUNCTIONS
# Initialises connection with the camera and undistorts frame  
# ------------------------------------------------------------------------

# Pre-computed HSV color ranges for better performance
LOWER_CREAM = np.array([90, 50, 50])
UPPER_CREAM = np.array([130, 255, 255])
LOWER_DARK = np.array([0, 0, 0])
UPPER_DARK = np.array([180, 255, 60])

# Model paths
contour_model_path = "models/YOLOv8/runs_chess/final_model_warp_contour/weights/best.pt"
colour_model_path = "models/YOLOv8/runs_chess/final_model_warp_colour/weights/best.pt"

# Load models
device = 0 if torch.cuda.is_available() else 'cpu'
contour_model = YOLO(contour_model_path)
colour_model = YOLO(colour_model_path)

# See what colour is in the current box
def check_colour(bbox_img):
    hsv = cv2.cvtColor(bbox_img, cv2.COLOR_BGR2HSV)
    
    # Create masks
    mask_cream = cv2.inRange(hsv, LOWER_CREAM, UPPER_CREAM)
    mask_dark = cv2.inRange(hsv, LOWER_DARK, UPPER_DARK)
    
    # Count non-zero pixels
    cream_pixels = cv2.countNonZero(mask_cream)
    dark_pixels = cv2.countNonZero(mask_dark)
    
    return 'white' if cream_pixels > dark_pixels else 'black'

# Get prediction from contour model
def contour_model_prediction(img):    
    result = contour_model.predict(
        source=img,
        conf=0.2,
        imgsz=448,
        device=device,
        verbose=False,
        half=True if device != 'cpu' else False  # Use FP16 on GPU for speed
    )
    return result[0]

# Get prediction from colour model
def colour_model_prediction(img):    
    result = colour_model.predict(
        source=img,
        conf=0.6,
        imgsz=448,
        device=device,
        verbose=False,
        half=True if device != 'cpu' else False  # Use FP16 on GPU for speed
    )
    return result[0]

def detect_pieces(warped):    
    if warped is None:
        return None, None, None, None, None, None, None
    
    # Prepare images - same preprocessing as with training
    contoured_warp = preprocess_image(warped)
    
    # Run predictions
    contour_predictions = contour_model_prediction(contoured_warp)
    colour_predictions = colour_model_prediction(warped)
    
    # Generate annotated images
    contour_annotated = contour_predictions.plot()
    colour_annotated = colour_predictions.plot()
    
    # Extract boxes from contour model
    contour_boxes = []
    contour_classes = []
    piece_colours = []
    
    for box in contour_predictions.boxes:
        # Extract coordinates as numpy array first
        xyxy = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = map(int, xyxy)
        
        class_id = int(box.cls[0].cpu().numpy())
        class_name = contour_predictions.names[class_id]
        
        contour_boxes.append({
            "x1": x1, "y1": y1,
            "x2": x2, "y2": y2
        })
        contour_classes.append(class_name)
        
        # Extract piece color
        boxed_region = warped[y1:y2, x1:x2]
        piece_colours.append(check_colour(boxed_region))
    
    # Extract boxes from colour model
    colour_boxes = []
    colour_classes = []
    
    for box in colour_predictions.boxes:
        # Extract coordinates as numpy array first
        xyxy = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = map(int, xyxy)
        
        class_id = int(box.cls[0].cpu().numpy())
        class_name = colour_predictions.names[class_id]
        
        colour_boxes.append({
            "x1": x1, "y1": y1,
            "x2": x2, "y2": y2
        })
        colour_classes.append(class_name)
    
    return contour_boxes, colour_boxes, contour_classes, colour_classes, contour_annotated, colour_annotated, piece_colours
