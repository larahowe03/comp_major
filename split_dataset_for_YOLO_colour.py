import os
import shutil
import random
import cv2
import numpy as np
from pathlib import Path

# -------------------------------
# CONFIGURATION
# -------------------------------
source_dir = Path("final_dataset")
output_dir = Path("final_dataset_yolo_warp_colour")

train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# -------------------------------
# OBJECT DETECTION FUNCTION
# -------------------------------
def detect_object_bbox(img_path):
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    
    h, w = img.shape[:2]

    # Define forbidden regions (x, y, width, height)
    forbidden_regions = [
        (0, 0, 40, 300),           # top-left
        (w - 200, h - 70, 200, 70) # bottom-right
    ]

    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define color ranges
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    lower_dark = np.array([0, 0, 0])
    upper_dark = np.array([180, 255, 60])
    
    # Create masks
    mask_cream = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_black = cv2.inRange(hsv, lower_dark, upper_dark)
    mask_combined = cv2.bitwise_or(mask_cream, mask_black)
    
    # Morphological cleanup
    kernel = np.ones((7, 7), np.uint8)
    mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(mask_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None

    def overlaps_forbidden(x, y, bw, bh):
        """Check if a bbox overlaps any forbidden region."""
        for fx, fy, fw, fh in forbidden_regions:
            # Check overlap (axis-aligned bounding boxes)
            if not (x + bw < fx or x > fx + fw or y + bh < fy or y > fy + fh):
                return True
        return False

    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 200:
            continue
        
        x, y, bw, bh = cv2.boundingRect(contour)
        if overlaps_forbidden(x, y, bw, bh):
            continue  # skip anything that touches forbidden region
        
        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        if (0.15 * w < cx < 0.85 * w) and (0.15 * h < cy < 0.85 * h):
            valid_contours.append(contour)
    
    if not valid_contours:
        # Fall back to largest non-forbidden contour, if any
        non_forbidden = [c for c in contours if not overlaps_forbidden(*cv2.boundingRect(c))]
        if non_forbidden:
            largest_contour = max(non_forbidden, key=cv2.contourArea)
        else:
            return None
    else:
        largest_contour = max(valid_contours, key=cv2.contourArea)
    
    # Bounding box
    x, y, box_w, box_h = cv2.boundingRect(largest_contour)
    padding = 5
    x = max(0, x - padding)
    y = max(0, y - padding)
    box_w = min(w - x, box_w + 2 * padding)
    box_h = min(h - y, box_h + 2 * padding)

    # Convert to YOLO format
    x_center = (x + box_w / 2) / w
    y_center = (y + box_h / 2) / h
    norm_width = box_w / w
    norm_height = box_h / h

    return (x_center, y_center, norm_width, norm_height), (x, y, box_w, box_h)


def make_clean_dir(path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

for split in ["train", "val", "test"]:
    make_clean_dir(output_dir / "images" / split)
    make_clean_dir(output_dir / "labels" / split)
    make_clean_dir(output_dir / "visualizations" / split)

class_names = sorted([d.name for d in source_dir.iterdir() if d.is_dir()])
class_to_id = {name: i for i, name in enumerate(class_names)}

print("🧩 Class mapping:")
for name, idx in class_to_id.items():
    print(f"{idx}: {name}")

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
            dest_lbl_dir = output_dir / "labels" / split_name
            dest_vis_dir = output_dir / "visualizations" / split_name
            
            dest_img_dir.mkdir(parents=True, exist_ok=True)
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

# -------------------------------
# CREATE data.yaml FILES
# -------------------------------
# Original images
yaml_path = output_dir / "data.yaml"
with open(yaml_path, "w") as f:
    f.write(f"train: /Users/lara.howe/Library/CloudStorage/OneDrive-Accenture/Documents/comp vision/major_project/final_dataset_yolo_warp_colour/images/train\n")
    f.write(f"val: /Users/lara.howe/Library/CloudStorage/OneDrive-Accenture/Documents/comp vision/major_project/final_dataset_yolo_warp_colour/images/val\n")
    f.write(f"test: /Users/lara.howe/Library/CloudStorage/OneDrive-Accenture/Documents/comp vision/major_project/final_dataset_yolo_warp_colour/images/test\n\n")
    f.write(f"nc: {len(class_names)}\n")
    f.write("names: [\n")
    for i, name in enumerate(class_names):
        comma = "," if i < len(class_names) - 1 else ""
        f.write(f"  '{name}'{comma}\n")
    f.write("]\n")