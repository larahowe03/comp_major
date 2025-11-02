import os
import shutil
import random
import cv2
import numpy as np
from pathlib import Path
from sklearn.linear_model import RANSACRegressor


# -------------------------------
# CONFIGURATION
# -------------------------------
source_dir = Path("big_chess_piece_dataset_png")
output_dir = Path("dataset_yolo_big")

train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1!"

# -------------------------------
# IMPROVED EDGE DETECTION METHODS
# -------------------------------

from sklearn.linear_model import RANSACRegressor

def detect_edges_shadow_free_ransac(im):
    """
    Enhanced Sobel + RANSAC filtering for removing shadow edges and outlier lines.
    """
    # [Same preprocessing as before]
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)
    background = cv2.GaussianBlur(filtered, (51,51), 0)
    illum_corrected = cv2.subtract(filtered, background)
    illum_corrected = cv2.normalize(illum_corrected, None, 0, 255, cv2.NORM_MINMAX)

    sobelx = cv2.Sobel(illum_corrected, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(illum_corrected, cv2.CV_64F, 0, 1, ksize=5)
    sobel_mag = np.sqrt(sobelx**2 + sobely**2)
    sobel_mag = cv2.normalize(sobel_mag, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    _, binary = cv2.threshold(sobel_mag, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological cleaning
    kernel = np.ones((2,2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # ---- RANSAC FILTERING ----
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    kept = []
    for c in contours:
        pts = c.squeeze()
        if len(pts.shape) < 2 or len(pts) < 20:
            continue
        x = pts[:,0].reshape(-1,1).astype(np.float32)
        y = pts[:,1].astype(np.float32)
        ransac = RANSACRegressor(residual_threshold=2.5, max_trials=100)
        ransac.fit(x, y)
        inlier_ratio = np.sum(ransac.inlier_mask_) / len(pts)
        if inlier_ratio > 0.5:  # tune threshold depending on your structure
            kept.append(c)
    
    filtered_edges = np.zeros_like(binary)
    cv2.drawContours(filtered_edges, kept, -1, 255, thickness=cv2.FILLED)
    return filtered_edges


def detect_edges_canny(im):
    """
    Apply Canny edge detection - often gives cleaner, more accurate edges.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    # Apply slight Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
    
    # Apply Canny edge detection with optimized thresholds
    # Lower threshold = 30, upper threshold = 100
    edges = cv2.Canny(blurred, 30, 100)
    
    # Optional: dilate slightly to make edges more visible
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    return edges


def detect_edges_adaptive(im):
    """
    Use adaptive thresholding combined with edge detection for better local contrast.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to reduce noise while keeping edges sharp
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Adaptive thresholding
    adaptive = cv2.adaptiveThreshold(
        filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Apply morphological operations to clean up
    kernel = np.ones((2, 2), np.uint8)
    adaptive = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Combine with Canny for better results
    edges_canny = cv2.Canny(filtered, 30, 100)
    
    # Combine both methods
    combined = cv2.bitwise_or(adaptive, edges_canny)
    
    return combined


def detect_edges_enhanced_sobel(im):
    """
    Enhanced Sobel with better preprocessing and post-processing.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Apply bilateral filter to reduce noise while preserving edges
    filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    # Apply Sobel in both directions with larger kernel for better edge detection
    sobelx = cv2.Sobel(filtered, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(filtered, cv2.CV_64F, 0, 1, ksize=5)
    
    # Compute gradient magnitude
    sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    # Normalize to 0-255
    sobel_magnitude = cv2.normalize(sobel_magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Use Otsu's thresholding for automatic threshold selection
    _, binary = cv2.threshold(sobel_magnitude, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Light morphological closing to connect nearby edges
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    return binary


def detect_edges_hybrid(im):
    """
    Hybrid approach combining multiple methods for best results.
    """
    # Get edges from different methods
    edges_canny = detect_edges_canny(im)
    edges_sobel = detect_edges_enhanced_sobel(im)
    
    # Combine using weighted average
    # Give more weight to Canny as it's generally more accurate
    combined = cv2.addWeighted(edges_canny, 0.6, edges_sobel, 0.4, 0)
    
    # Threshold the combined result
    _, final = cv2.threshold(combined, 50, 255, cv2.THRESH_BINARY)
    
    return final


def detect_edges_laplacian(im):
    """
    Use Laplacian edge detection for finding edges in all directions.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
    
    # Apply Laplacian
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F, ksize=5)
    
    # Convert to absolute values and normalize
    laplacian = np.absolute(laplacian)
    laplacian = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Threshold
    _, binary = cv2.threshold(laplacian, 30, 255, cv2.THRESH_BINARY)
    
    # Clean up with morphology
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    return binary


def preprocess_image(img, method='canny'):
    """
    Apply full preprocessing pipeline with selectable edge detection method.
    
    Methods available:
    - 'canny': Canny edge detection (recommended for clean edges)
    - 'adaptive': Adaptive thresholding + Canny (good for varying lighting)
    - 'sobel': Enhanced Sobel (good for gradient-based detection)
    - 'hybrid': Combination of Canny and Sobel (balanced)
    - 'laplacian': Laplacian edge detection (omnidirectional)
    """
    # Resize to 1/3 of original size
    h, w = img.shape[:2]
    img_resized = cv2.resize(img, (w // 3, h // 3), interpolation=cv2.INTER_AREA)
    
    # Select edge detection method
    if method == 'canny':
        edges = detect_edges_canny(img_resized)
    elif method == 'adaptive':
        edges = detect_edges_adaptive(img_resized)
    elif method == 'sobel':
        edges = detect_edges_enhanced_sobel(img_resized)
    elif method == 'hybrid':
        edges = detect_edges_hybrid(img_resized)
    elif method == 'laplacian':
        edges = detect_edges_laplacian(img_resized)
    elif method == 'shadow':
        edges = detect_edges_shadow_free_ransac(img_resized)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Convert back to BGR for consistency
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    return edges_bgr


# -------------------------------
# OBJECT DETECTION FUNCTION
# -------------------------------
def detect_object_bbox(img_path):
    """
    Detect cream/black chess piece on grey background.
    Cream pieces: muted yellow #8C7C48
    Background: grey #9E9D99
    Returns normalized (x_center, y_center, width, height) for YOLO format.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    
    h, w = img.shape[:2]
    
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define color ranges
    # For muted yellow/cream pieces (#8C7C48 - yellowish brown)
    lower_cream = np.array([15, 20, 50])   # Yellow-brown hue
    upper_cream = np.array([35, 150, 200])
    
    # For black pieces (dark colors)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 80])  # Very dark values
    
    # Create masks
    mask_cream = cv2.inRange(hsv, lower_cream, upper_cream)
    mask_black = cv2.inRange(hsv, lower_black, upper_black)
    
    # Combine masks (either cream OR black pieces)
    mask_combined = cv2.bitwise_or(mask_cream, mask_black)
    
    # Morphological operations to clean up and connect the chess piece
    kernel = np.ones((7, 7), np.uint8)
    mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(mask_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Filter contours by area and position (should be in middle region)
    valid_contours = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        # Filter out very small noise
        if area < 200:
            continue
            
        # Get contour center
        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        # Check if roughly in the middle region (within middle 80% of image)
        if (0.15 * w < cx < 0.85 * w) and (0.15 * h < cy < 0.85 * h):
            valid_contours.append(contour)
    
    if not valid_contours:
        # If no valid contours in middle, just take the largest one
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
        else:
            return None
    else:
        # Get the largest valid contour (the chess piece)
        largest_contour = max(valid_contours, key=cv2.contourArea)
    
    # Get bounding box
    x, y, box_w, box_h = cv2.boundingRect(largest_contour)
    
    # Add some padding to the bounding box
    padding = 5
    x = max(0, x - padding)
    y = max(0, y - padding)
    box_w = min(w - x, box_w + 2 * padding)
    box_h = min(h - y, box_h + 2 * padding)
    
    # Convert to YOLO format (normalized center coordinates and dimensions)
    x_center = (x + box_w / 2) / w
    y_center = (y + box_h / 2) / h
    norm_width = box_w / w
    norm_height = box_h / h
    
    return (x_center, y_center, norm_width, norm_height), (x, y, box_w, box_h)


# -------------------------------
# MAKE CLEAN DIRECTORIES
# -------------------------------
def make_clean_dir(path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

for split in ["train", "val", "test"]:
    make_clean_dir(output_dir / "images" / split)
    make_clean_dir(output_dir / "images_preprocessed" / split)
    make_clean_dir(output_dir / "labels" / split)
    make_clean_dir(output_dir / "visualizations" / split)

# -------------------------------
# CLASS NAME MAPPING
# -------------------------------
class_names = sorted([d.name for d in source_dir.iterdir() if d.is_dir()])
class_to_id = {name: i for i, name in enumerate(class_names)}

print("🧩 Class mapping:")
for name, idx in class_to_id.items():
    print(f"{idx}: {name}")

# -------------------------------
# EDGE DETECTION METHOD SELECTION
# -------------------------------
EDGE_METHOD = 'shadow'  # Options: 'canny', 'adaptive', 'sobel', 'hybrid', 'laplacian'
print(f"\n🎨 Using edge detection method: {EDGE_METHOD.upper()}")

# -------------------------------
# SPLIT + GENERATE LABELS
# -------------------------------
for class_dir in source_dir.iterdir():
    if not class_dir.is_dir():
        continue

    class_id = class_to_id[class_dir.name]
    images = [f for f in class_dir.glob("*.*") if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    random.shuffle(images)

    n_total = len(images)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)

    splits = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:]
    }

    for split_name, split_files in splits.items():
        for img_path in split_files:
            dest_img_dir = output_dir / "images" / split_name
            dest_prep_dir = output_dir / "images_preprocessed" / split_name
            dest_lbl_dir = output_dir / "labels" / split_name
            dest_vis_dir = output_dir / "visualizations" / split_name
            
            dest_img_dir.mkdir(parents=True, exist_ok=True)
            dest_prep_dir.mkdir(parents=True, exist_ok=True)
            dest_lbl_dir.mkdir(parents=True, exist_ok=True)
            dest_vis_dir.mkdir(parents=True, exist_ok=True)

            # Read original image
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"⚠️  Could not read {img_path.name}, skipping")
                continue

            # Detect bounding box on original image
            bbox_result = detect_object_bbox(img_path)
            
            if bbox_result is None:
                print(f"⚠️  Could not detect object in {img_path.name}, using default bbox")
                yolo_bbox = (0.5, 0.65, 0.25, 0.45)
                pixel_bbox = None
            else:
                yolo_bbox, pixel_bbox = bbox_result

            # Copy original image
            shutil.copy(img_path, dest_img_dir / img_path.name)

            # Apply preprocessing and save
            preprocessed_img = preprocess_image(img, method=EDGE_METHOD)
            cv2.imwrite(str(dest_prep_dir / img_path.name), preprocessed_img)

            # Create YOLO label file
            label_path = dest_lbl_dir / f"{img_path.stem}.txt"
            with open(label_path, "w") as f:
                x_c, y_c, bbox_w, bbox_h = yolo_bbox
                f.write(f"{class_id} {x_c:.6f} {y_c:.6f} {bbox_w:.6f} {bbox_h:.6f}\n")

            # Create visualization with bounding box on original image
            if pixel_bbox is not None:
                vis_img = img.copy()
                x, y, box_w, box_h = pixel_bbox
                cv2.rectangle(vis_img, (x, y), (x + box_w, y + box_h), (0, 255, 0), 2)
                
                # Add class label
                label_text = f"{class_dir.name}"
                cv2.putText(vis_img, label_text, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                cv2.imwrite(str(dest_vis_dir / img_path.name), vis_img)

    print(f"✅ {class_dir.name}: {n_total} images split, preprocessed, and labeled with bboxes.")

print("\n🎯 YOLO dataset with auto-detected bounding boxes created at:")
print(f"📁 {output_dir}/images/  (original images)")
print(f"📁 {output_dir}/images_preprocessed/  (edge-detected images using {EDGE_METHOD.upper()})")
print(f"📁 {output_dir}/labels/")
print(f"📁 {output_dir}/visualizations/  (images with bboxes drawn)")

# -------------------------------
# CREATE data.yaml FILES
# -------------------------------
# Original images
yaml_path = output_dir / "data.yaml"
with open(yaml_path, "w") as f:
    f.write(f"train: {output_dir}/images/train\n")
    f.write(f"val: {output_dir}/images/val\n")
    f.write(f"test: {output_dir}/images/test\n\n")
    f.write(f"nc: {len(class_names)}\n")
    f.write("names: [\n")
    for i, name in enumerate(class_names):
        comma = "," if i < len(class_names) - 1 else ""
        f.write(f"  '{name}'{comma}\n")
    f.write("]\n")

# Preprocessed images
yaml_path_prep = output_dir / "data_preprocessed.yaml"
with open(yaml_path_prep, "w") as f:
    f.write(f"train: {output_dir}/images_preprocessed/train\n")
    f.write(f"val: {output_dir}/images_preprocessed/val\n")
    f.write(f"test: {output_dir}/images_preprocessed/test\n\n")
    f.write(f"nc: {len(class_names)}\n")
    f.write("names: [\n")
    for i, name in enumerate(class_names):
        comma = "," if i < len(class_names) - 1 else ""
        f.write(f"  '{name}'{comma}\n")
    f.write("]\n")

print(f"\n🧾 data.yaml files generated at:")
print(f"   {yaml_path} (for original images)")
print(f"   {yaml_path_prep} (for preprocessed images)")