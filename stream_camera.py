import cv2
import pickle
from warp_board import process_chess_image
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
from mask_out import mask_out

from ultralytics import YOLO
import torch
from pathlib import Path
import os

def get_pieces(img):
    model_path = "runs_chess/chess_yolov8/weights/best.pt"

    # Load YOLO
    device = 0 if torch.cuda.is_available() else 'cpu'
    model = YOLO(model_path)

    # Check if image is grayscale and convert to BGR
    if len(img.shape) == 2:  # Grayscale (H, W)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 1:  # Grayscale (H, W, 1)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Run predictions
    results = model.predict(
        source=img,
        conf=0.1,
        imgsz=448,
        device=device,
        save=False,
        verbose=False
    )

    # Get result
    result = results[0]
    annotated_img = result.plot()
    
    # Optional: extract detections
    detections = []
    for box in result.boxes:
        cls_id = int(box.cls)
        conf = float(box.conf)
        class_name = result.names[cls_id]
        bbox = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
        
        detections.append({
            'class': class_name,
            'confidence': conf,
            'bbox': bbox
        })
    
    return annotated_img    

def detect_edges(im):
    # Convert to grayscale
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    # Define the aggressive edge detection kernel
    aggressive_kernel = np.array([
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]
    ], dtype=np.float32)
    
    # Apply the kernel using filter2D
    edges = cv2.filter2D(gray, -1, aggressive_kernel)
    
    # Normalize to 0-255 range
    edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Threshold to make edges white, everything else black
    _, binary = cv2.threshold(edges, 30, 255, cv2.THRESH_BINARY)
    
    # Remove straight lines using HoughLinesP
    lines = cv2.HoughLinesP(binary, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
    
    # Create a mask to remove detected lines
    line_mask = np.ones_like(binary) * 255
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Draw thick lines on the mask to remove them
            cv2.line(line_mask, (x1, y1), (x2, y2), 0, thickness=3)
    
    # Apply the mask to remove lines
    binary = cv2.bitwise_and(binary, line_mask)
    
    return binary

# def contour_detect(im):
#     # Convert to grayscale
#     gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
#     # Apply bilateral filter to reduce noise while keeping edges sharp
#     filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    
#     # Apply CLAHE for better contrast
#     # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     # enhanced = clahe.apply(filtered)
    
#     # Adaptive thresholding for better edge detection
#     binary = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                    cv2.THRESH_BINARY, 11, 2)
    
#     # Morphological operations to clean up
#     kernel = np.ones((3, 3), np.uint8)
#     morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
#     morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
    
#     # Edge detection with optimized parameters
#     edges = cv2.Canny(morph, 50, 150)
    
#     # Find contours
#     contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     # Intelligent filtering
#     filtered_contours = []
#     for c in contours:
#         area = cv2.contourArea(c)
#         perimeter = cv2.arcLength(c, True)
        
#         # Filter by area and aspect ratio
#         if area > 100:
#             x, y, w, h = cv2.boundingRect(c)
#             aspect_ratio = float(w) / h if h > 0 else 0
            
#             # Keep contours with reasonable aspect ratios (not too thin)
#             if 0.2 < aspect_ratio < 5.0:
#                 filtered_contours.append(c)
    
#     # Draw contours with different colors based on size
#     result = im.copy()
#     for c in filtered_contours:
#         area = cv2.contourArea(c)
#         # Color code by size: small=green, medium=blue, large=red
#         if area < 500:
#             color = (0, 255, 0)  # Green
#         elif area < 2000:
#             color = (255, 0, 0)  # Blue
#         else:
#             color = (0, 0, 255)  # Red
        
#         cv2.drawContours(result, [c], -1, color, 2)
    
#     return result

def mask_black(im):
    # Convert to grayscale
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    # Find black pixels (below threshold)
    mask = gray < 70
    mask = mask.astype(np.uint8) * 255  # Convert boolean to 0-255
    
    # Morphological operations to clean up
    # kernel = np.ones((5, 5), np.uint8)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Find contours in the mask
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours on the original image
    h, w = im.shape[:2]
    result = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Filter contours by area
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:  # Minimum area threshold
            # Draw contour

            cv2.drawContours(result, contours, -1, (255, 255, 255), 2)  # Green contours
            
            # Optional: draw bounding box
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue box
    
    return result
def mask_white(im):
    # Convert to HSV
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    
    # Define range for cream/off-white colors
    lower_cream = np.array([0, 0, 140])
    upper_cream = np.array([40, 120, 200])
    
    # Create mask for cream/white pixels
    mask = cv2.inRange(hsv, lower_cream, upper_cream)
    
    # Morphological closing to remove noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Optional: opening to remove small blue spots
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Replace white/cream with blue (BGR format)
    result = im.copy()
    result[mask > 0] = [255, 0, 0]  # Blue in BGR
    
    return result
# def colour_space(im):
#     # Convert to HSV
#     im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
#     H, S, V = cv2.split(im_hsv)
    
#     # Prepare pixel colors for visualization
#     height, width = im.shape[:2]
#     pixel_colours = im.reshape((height * width, 3)) / 255.0  # Normalize to 0-1
#     pixel_colours = pixel_colours[:, ::-1]  # BGR to RGB
    
#     # Enable interactive mode
#     plt.ion()
    
#     # Create or clear figure
#     plt.clf()
#     fig = plt.gcf()
#     fig.set_size_inches(12, 10)
#     axis = fig.add_subplot(1, 1, 1, projection="3d")
    
#     # Downsample for performance
#     step = max(1, (height * width) // 10000)
    
#     axis.scatter(H.flatten()[::step], 
#                  S.flatten()[::step], 
#                  V.flatten()[::step], 
#                  c=pixel_colours[::step], 
#                  marker='.', 
#                  s=1,
#                  alpha=0.5)
    
#     axis.set_xlabel("Hue", fontsize=12)
#     axis.set_ylabel("Saturation", fontsize=12)
#     axis.set_zlabel("Value", fontsize=12)
#     axis.set_title("HSV Color Space", fontsize=14)
#     axis.view_init(elev=30, azim=45)
    
#     plt.tight_layout()
#     plt.draw()
#     plt.pause(0.001)  # Small pause to update display

def adjust_squares(img):
    """
    Brighten white squares and darken black squares.
    Black square is at top-right (alternating pattern).
    """
    h, w = img.shape[:2]
    cell_h = h // 8
    cell_w = w // 8
    
    result = img.copy()
    
    for row in range(8):
        for col in range(8):
            # Calculate if this square should be black or white
            # Top-right is black, so pattern is: (row + col) % 2 == 1 for black
            is_black_square = (row + col) % 2 == 1
            
            # Get cell coordinates
            y1 = row * cell_h
            y2 = (row + 1) * cell_h
            x1 = col * cell_w
            x2 = (col + 1) * cell_w
            
            # Extract cell
            cell = result[y1:y2, x1:x2].copy()
            
            if is_black_square:
                # Darken black squares (gamma > 1)
                cell = apply_gamma_correction(cell, gamma=1.5)
            else:
                # Brighten white squares (gamma < 1)
                cell = apply_gamma_correction(cell, gamma=0.7)
            
            # Put cell back
            result[y1:y2, x1:x2] = cell
    
    return result

def clahethis(img):
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # Split into L, A, B channels
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    
    # Merge channels back
    lab_clahe = cv2.merge([l_clahe, a, b])
    
    # Convert back to BGR
    img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    
    return img_clahe

def undistort(img, K, d):
    return cv2.undistort(img, K, d, None, K)

def apply_gamma_correction(img, gamma=1.5):
    """
    Apply gamma correction to brighten/darken image
    gamma < 1: brighten
    gamma > 1: darken
    gamma = 1: no change
    """
    # Build lookup table
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 
                      for i in range(256)]).astype("uint8")
    
    # Apply gamma correction using lookup table
    return cv2.LUT(img, table)

if __name__ == "__main__":
    # Replace with YOUR phone's IP and port from the DroidCam app
    phone_ip = "10.19.204.143"
    port = "4747"

    # DroidCam streaming URLs - try these in order:
    urls = [
        f"http://{phone_ip}:{port}/video",
        f"http://{phone_ip}:{port}/mjpegfeed",
    ]

    cap = None
    for url in urls:
        print(f"Trying: {url}")
        cap = cv2.VideoCapture(url)
        if cap.isOpened():
            print(f"Connected successfully to {url}")
            break
        cap.release()

    if not cap or not cap.isOpened():
        print("Could not connect. Check:")
        print("1. Phone and laptop on same WiFi")
        print("2. IP address is correct")
        print("3. DroidCam app is running")
        exit()

    with open("calibration_coefficients.pkl", "rb") as f:
        data = pickle.load(f)

    K = data['camera_matrix']
    d = data['distortion_coeffs']

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Connection lost")
            break

        frame_undistorted = undistort(frame, K, d)

        processed, blah = process_chess_image(frame_undistorted)
        if processed is None:
            continue
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        processed = apply_gamma_correction(processed)

        adjusted = adjust_squares(processed)

        
        # clahed = clahethis(processed)

        blue = cv2.GaussianBlur(processed, (7, 7), 0)

        edge = detect_edges(blue)

        dedsf = get_pieces(edge)

        # colour_space(clahed)

        wgite = mask_white(processed)
        nblhac = mask_black(processed)

        img = contour_detect(processed)
        masekd = mask_out(processed)

        black_detect = get_pieces(nblhac)

        if ret:
            cv2.imshow('Processed', processed)
            # cv2.imwrite("blah.png", processed)
            # yolod = resynet_test(processed)
        

        cv2.imshow('Undistorted', frame_undistorted)
        cv2.imshow('adjusted', adjusted)
        cv2.imshow('blah', blah)
        cv2.imshow('Distorted', frame)
        # cv2.imshow('clahed', clahed)
        cv2.imshow('black_detect', black_detect)
        cv2.imshow('edge', edge)
        cv2.imshow('wgite', wgite)
        cv2.imshow('nblhac', nblhac)
        cv2.imshow('dedsf', dedsf)
        cv2.imshow('img', img)
        cv2.imshow('masekd', masekd)
        # cv2.imshow('hsv', hsv)
        # cv2.imshow('thresh', thresh)
        # if yolod is not None and yolod.size > 0:
        #     cv2.imshow('yolod', yolod)
        # else:
        #     print("⚠️ Skipping frame — yolod is empty or invalid.")

        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

