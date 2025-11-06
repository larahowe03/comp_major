import cv2
import pickle
import numpy as np
import torch
from ultralytics import YOLO
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

# Pre-compute HSV color ranges for better performance
LOWER_CREAM = np.array([90, 50, 50])
UPPER_CREAM = np.array([130, 255, 255])
LOWER_DARK = np.array([0, 0, 0])
UPPER_DARK = np.array([180, 255, 60])

def check_colour(bbox_img):
    """Optimized color detection with pre-computed ranges"""
    hsv = cv2.cvtColor(bbox_img, cv2.COLOR_BGR2HSV)
    
    # Create masks
    mask_cream = cv2.inRange(hsv, LOWER_CREAM, UPPER_CREAM)
    mask_dark = cv2.inRange(hsv, LOWER_DARK, UPPER_DARK)
    
    # Count non-zero pixels
    cream_pixels = cv2.countNonZero(mask_cream)
    dark_pixels = cv2.countNonZero(mask_dark)
    
    return 'white' if cream_pixels > dark_pixels else 'black'

def contour_model_prediction(img):
    """Optimized contour model prediction"""
    result = contour_model.predict(
        source=img,
        conf=0.2,
        imgsz=448,
        device=device,
        verbose=False,
        half=True if device != 'cpu' else False  # Use FP16 on GPU for speed
    )
    return result[0]

def make_clahe(img):
    im2 = img.copy()

    # Convert to LAB color space
    lab = cv2.cvtColor(im2, cv2.COLOR_BGR2LAB)

    # Split into L, A, B channels
    l, a, b = cv2.split(lab)

    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)

    # Merge channels back
    lab_clahe = cv2.merge([l_clahe, a, b])

    # Convert back to BGR
    im2 = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    return im2

def colour_model_prediction(img):
    # im2 = make_clahe(img)
    """Optimized colour model prediction"""
    result = colour_model.predict(
        source=img,
        conf=0.6,
        imgsz=448,
        device=device,
        verbose=False,
        half=True if device != 'cpu' else False  # Use FP16 on GPU for speed
    )
    return result[0]

def undistort(img, K, d):
    """Optimized undistortion"""
    return cv2.undistort(img, K, d, None, K)

# Global variables
camera = None
K_matrix = None
dist_coeffs = None

def initialize_camera(phone_ip="10.16.241.228", port="4747"):
    """Initialize camera connection with optimized settings"""
    urls = [
        f"http://{phone_ip}:{port}/video",
        f"http://{phone_ip}:{port}/mjpegfeed",
    ]
    
    for url in urls:
        print(f"Trying: {url}")
        cap = cv2.VideoCapture(url)
        
        if cap.isOpened():
            # Set buffer size to reduce latency
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            print(f"Connected successfully to {url}")
            return cap
        cap.release()
    
    print("Could not connect to camera")
    return None

def load_calibration(filepath="calibration_coefficients.pkl"):
    """Load camera calibration data"""
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data['camera_matrix'], data['distortion_coeffs']

def init_detection_system():
    """Initialize the detection system (call once at startup)"""
    global camera, K_matrix, dist_coeffs
    
    camera = initialize_camera()
    if camera is None:
        return False
    
    K_matrix, dist_coeffs = load_calibration()
    
    # Warm up models (first inference is slower)
    dummy_img = np.zeros((448, 448, 3), dtype=np.uint8)
    contour_model.predict(dummy_img, verbose=False)
    colour_model.predict(dummy_img, verbose=False)
    
    return True

def get_current_frame():
    """Get and process current frame from camera"""
    global camera, K_matrix, dist_coeffs
    
    if camera is None:
        return None, None, None, None
    
    ret, frame = camera.read()
    if not ret:
        return None, None, None, None
    
    # Undistort
    undistorted = undistort(frame, K_matrix, dist_coeffs)
    
    # Warp board
    warp_margined, warp_unmargined, contoured_img, pts_src = process_chess_image(undistorted)
    
    if warp_margined is None:
        return undistorted, undistorted, None, None
    
    return warp_margined, warp_unmargined, contoured_img, pts_src

def detect_pieces(warped):
    """Optimized piece detection with reduced redundancy"""
    
    if warped is None:
        return None, None, None, None, None, None, None
    
    # Prepare images
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

def cleanup_camera():
    """Clean up camera resources"""
    global camera
    if camera is not None:
        camera.release()
        cv2.destroyAllWindows()