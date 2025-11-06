import cv2
import numpy as np
import pickle
from chess_detection import contour_model, colour_model
from src.main.warp_board import process_chess_image

# ------------------------------------------------------------------------
# CAMERA SETUP
# Initialises connection with the camera and undistorts frame  
# ------------------------------------------------------------------------

def undistort(img, K, d):
    """
    Undistortion of the camera frame
    """
    
    return cv2.undistort(img, K, d, None, K)


def initialize_camera(phone_ip="10.16.241.228", port="4747"):
    """
    Initialize camera connection
    """
    
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
    """
    Load camera calibration data
    """
    
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data['camera_matrix'], data['distortion_coeffs']


def init_detection_system():
    """
    Initialize the detection system (call once at startup)
    """
    
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


def cleanup_camera():
    """
    Clean up camera resources
    """
    
    global camera
    if camera is not None:
        camera.release()
        cv2.destroyAllWindows()
        

def get_current_frame():
    """
    Get and process current frame from camera
    """
    
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