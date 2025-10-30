import numpy as np
import cv2
import os

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
    
    # Morphological operations to connect nearby edges
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Remove small connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    # Filter by area - keep only large components
    min_area = 100  # Adjust this threshold based on your needs
    filtered = np.zeros_like(binary)
    
    for i in range(1, num_labels):  # Skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            filtered[labels == i] = 255
    
    return filtered

for img_name in os.listdir("dataset_yolo/images/train"):
    img = cv2.imread(f"dataset_yolo/images/train/{img_name}")
    
    # Apply Gaussian blur before resizing
    img_blurred = cv2.GaussianBlur(img, (31, 31), 0)
    
    # Resize to 1/4 of original size
    h, w = img_blurred.shape[:2]
    img_resized = cv2.resize(img_blurred, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    
    # Detect edges on resized image
    contoured = detect_edges(img_resized)
    
    cv2.imwrite(f"dataset_yolo/images/train_contour/{img_name}", contoured)