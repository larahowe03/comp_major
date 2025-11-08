import cv2
import numpy as np
import torch
from ultralytics import YOLO
# from src.training.split_dataset_for_YOLO_contour import preprocess_image

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

print(f"Using device: {device}")

def detect_edges_laplacian(im):
    # Convert to HSV for better color masking
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    
    # Create masks for colors to remove
    # Blue mask (wider range to catch various blues)
    lower_blue = np.array([90, 50, 50])    # Hue ~90-130 is blue
    upper_blue = np.array([130, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Dark brown/black mask (low value/brightness)
    lower_dark = np.array([0, 0, 0])
    upper_dark = np.array([180, 255, 60])  # Very low brightness (V channel)
    mask_dark = cv2.inRange(hsv, lower_dark, upper_dark)
    
    # Brown mask (orange-brown hues)
    lower_brown = np.array([5, 30, 30])
    upper_brown = np.array([25, 255, 150])
    mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)
    
    # Combine all masks (OR operation - mask out any of these colors)
    combined_mask = cv2.bitwise_or(mask_blue, mask_dark)
    combined_mask = cv2.bitwise_or(combined_mask, mask_brown)
        
    # Apply morphological operations to clean up the color mask
    kernel_clean = np.ones((5, 5), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_clean, iterations=2)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_clean, iterations=1)
    
    # Convert to grayscale
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    # Apply the color mask to the grayscale image
    gray_masked = cv2.bitwise_and(gray, gray, mask=combined_mask)
    
    # Apply Laplacian on masked image
    laplacian = cv2.Laplacian(gray_masked, cv2.CV_64F, ksize=5)
    
    # Convert to absolute values and normalize
    laplacian = np.absolute(laplacian)
    laplacian = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Threshold
    _, binary = cv2.threshold(laplacian, 30, 255, cv2.THRESH_BINARY)
    
    # Apply color mask again to ensure masked regions stay removed
    binary = cv2.bitwise_and(binary, binary, mask=combined_mask)
    
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    return binary

def preprocess_image(img):    
    # Select edge detection method
    edges = detect_edges_laplacian(img)
    
    # Convert back to BGR for consistency
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    return edges_bgr


def check_colour(bbox_img):
    """
    Color detection with pre-computed ranges
    """
    
    hsv = cv2.cvtColor(bbox_img, cv2.COLOR_BGR2HSV)
    
    # Create masks
    mask_cream = cv2.inRange(hsv, LOWER_CREAM, UPPER_CREAM)
    mask_dark = cv2.inRange(hsv, LOWER_DARK, UPPER_DARK)
    
    # Count non-zero pixels
    cream_pixels = cv2.countNonZero(mask_cream)
    dark_pixels = cv2.countNonZero(mask_dark)
    
    return 'white' if cream_pixels > dark_pixels else 'black'


def contour_model_prediction(img):
    """
    Contour model prediction
    """
    
    result = contour_model.predict(
        source=img,
        conf=0.2,
        imgsz=448,
        device=device,
        verbose=False,
        half=True if device != 'cpu' else False  # Use FP16 on GPU for speed
    )
    return result[0]


def colour_model_prediction(img):
    """
    Colour model prediction
    """
    
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
    """
    Piece detection 
    """
    
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
        # Extract coordinates as numpy array first (faster)
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
        # Extract coordinates as numpy array first (faster)
        xyxy = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = map(int, xyxy)
        
        class_id = int(box.cls[0].cpu().numpy())
        class_name = colour_predictions.names[class_id]
        
        colour_boxes.append({
            "x1": x1, "y1": y1,
            "x2": x2, "y2": y2
        })
        colour_classes.append(class_name)
    
    return (contour_boxes, colour_boxes, contour_classes, colour_classes, 
            contour_annotated, colour_annotated, piece_colours)
