import os
import shutil
import random
import cv2
import numpy as np
from pathlib import Path

# -------------------------------
# CONFIGURATION
# -------------------------------
source_dir = Path("chess_piece_images_for_ResNet")
output_dir = Path("dataset_yolo")

train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1!"

# -------------------------------
# EDGE DETECTION PREPROCESSING
# -------------------------------
def detect_edges(im):
    """
    Apply Sobel edge detection preprocessing to image with more detail.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    # Apply Sobel edge detection in X and Y directions
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Compute gradient magnitude
    sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    # Normalize to 0-255 range
    sobel_magnitude = cv2.normalize(sobel_magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Lower threshold to capture more details and subtle edges
    _, binary = cv2.threshold(sobel_magnitude, 20, 255, cv2.THRESH_BINARY)
    
    # Skip or minimize morphological operations to preserve detail
    # kernel = np.ones((2, 2), np.uint8)
    # binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Remove only very small connected components (keep more edges)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    # Filter by area - keep more edges by lowering thresholds
    min_area = 20    # Lower to keep more detail lines
    max_area = 5000  # Keep upper limit
    filtered = np.zeros_like(binary)
    
    for i in range(1, num_labels):  # Skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            filtered[labels == i] = 255
    
    return filtered

def preprocess_image(img):
    """
    Apply full preprocessing pipeline: blur, resize, and edge detection.
    """
    # Apply lighter Gaussian blur to preserve more detail
    img_blurred = cv2.GaussianBlur(img, (15, 15), 0)  # Reduced from (31, 31)
    
    # Resize to 1/4 of original size
    h, w = img_blurred.shape[:2]
    img_resized = cv2.resize(img_blurred, (w // 3, h // 3), interpolation=cv2.INTER_AREA)
    
    # Detect edges on resized image
    contoured = detect_edges(img_resized)
    
    # Convert back to BGR for consistency
    contoured_bgr = cv2.cvtColor(contoured, cv2.COLOR_GRAY2BGR)
    
    return contoured_bgr

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
    make_clean_dir(output_dir / "images_preprocessed" / split)  # New: preprocessed images
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
            preprocessed_img = preprocess_image(img)
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
print(f"📁 {output_dir}/images_preprocessed/  (Sobel edge-detected images)")
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