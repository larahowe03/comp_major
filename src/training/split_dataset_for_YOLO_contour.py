import os
import shutil
import random
import cv2
import numpy as np
from pathlib import Path

source_dir = Path("../final_dataset")
output_dir = Path("datasets/dataset_yolo_warp_contour")

train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

def get_mask(hsv_im):
    # Cream mask for white pieces
    lower_cream = np.array([90, 50, 50]) 
    upper_cream = np.array([130, 255, 255])
    mask_cream = cv2.inRange(hsv_im, lower_cream, upper_cream)
    
    # Dark brown/black for black pieces
    lower_dark = np.array([0, 0, 0])
    upper_dark = np.array([180, 255, 60])
    mask_dark = cv2.inRange(hsv_im, lower_dark, upper_dark)
    
    # Brown mask for properly including all black pieces
    lower_brown = np.array([5, 30, 30])
    upper_brown = np.array([25, 255, 150])
    mask_brown = cv2.inRange(hsv_im, lower_brown, upper_brown)
    
    # Combine all masks
    combined_mask = cv2.bitwise_or(mask_cream, mask_dark)
    combined_mask = cv2.bitwise_or(combined_mask, mask_brown)
        
    # Morphological operation to clean mask
    kernel_clean = np.ones((5, 5), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_clean, iterations=2)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_clean, iterations=1)

    return combined_mask

def detect_edges_laplacian(im):
    # Convert to HSV for better color masking
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    combined_mask = get_mask(hsv)
    
    # Convert to grayscale
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    # Apply the color mask to the grayscale image
    gray_masked = cv2.bitwise_and(gray, gray, mask=combined_mask)
    
    # Apply Laplacian on masked image
    laplacian = cv2.Laplacian(gray_masked, cv2.CV_64F, ksize=5)
    
    # Convert to absolute values and normalise
    laplacian = np.absolute(laplacian)
    laplacian = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Threshold
    _, binary = cv2.threshold(laplacian, 30, 255, cv2.THRESH_BINARY)
    
    # Apply color mask again to ensure masked regions stay removed
    binary = cv2.bitwise_and(binary, binary, mask=combined_mask)
    
    # Cleaning up the contour result again
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    return binary

def preprocess_image(img):    
    edges = detect_edges_laplacian(img)
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return edges_bgr


# Function to get the bounding box needed for yolo
def detect_object_bbox(img_path):
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    
    h, w = img.shape[:2]
    
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    combined_mask = get_mask(hsv)
        
    # Find contours
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Filter contours by area and position (should be in middle region)
    valid_contours = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        # Ignore reall small contours as bounding boxes
        if area < 200:
            continue
            
        # Get contour center
        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue
        contour_x = int(M["m10"] / M["m00"])
        contour_y = int(M["m01"] / M["m00"])
        
        # Check if roughly in the middle region (within middle 80% of image)
        if (0.15 * w < contour_x < 0.85 * w) and (0.15 * h < contour_y < 0.85 * h):
            valid_contours.append(contour)
    
    # If this doesnt work just take the largest contour
    if not valid_contours:
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
        else:
            return None
    else:
        # Get the largest valid contour (the chess piece)
        largest_contour = max(valid_contours, key=cv2.contourArea)
    
    # Get bounding box
    x, y, box_w, box_h = cv2.boundingRect(largest_contour)
    
    # Add 5px of padding
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

if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        os.mkdir(f"{output_dir}/images/{split}")
        os.mkdir(f"{output_dir}/labels/{split}")
        os.mkdir(f"{output_dir}/visualisations/{split}")

    # Get all class directories
    class_dirs = sorted([d for d in source_dir.iterdir() if d.is_dir()])

    # Extract second word from directory names as contour is just the chess piece type not the colour too
    class_names = sorted(list(set([d.name.split('_')[1] for d in class_dirs])))
    class_to_id = {name: i for i, name in enumerate(class_names)}

    for class_dir in class_dirs:
        # Extract the piece type (second word after underscore)
        piece_type = class_dir.name.split('_')[1]
        class_id = class_to_id[piece_type]
        
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
                dest_img_dir = f"{output_dir}/images/{split_name}"
                dest_lbl_dir = f"{output_dir}/labels/{split_name}"
                dest_vis_dir = f"{output_dir}/visualisations/{split_name}"
                
                os.mkdir(dest_img_dir)
                os.mkdir(dest_lbl_dir)
                os.mkdir(dest_vis_dir)

                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                # Detect bounding box on original image
                bbox_result = detect_object_bbox(img_path)
                
                if bbox_result is None:
                    print(f"No bounding box found in {img_path.name}, using default bbox")
                    yolo_bbox = (0.5, 0.65, 0.25, 0.45)
                    pixel_bbox = None
                else:
                    yolo_bbox, pixel_bbox = bbox_result

                # Copy original image
                shutil.copy(img_path, f"{dest_img_dir}/{img_path.name}")

                # Apply preprocessing and save
                preprocessed_img = preprocess_image(img)
                cv2.imwrite(f"{dest_img_dir}/{img_path.name}", preprocessed_img)

                # Create YOLO label file
                label_path = f"{dest_lbl_dir}/{img_path.stem}.txt"
                with open(label_path, "w") as f:
                    x_c, y_c, bbox_w, bbox_h = yolo_bbox
                    f.write(f"{class_id} {x_c:.6f} {y_c:.6f} {bbox_w:.6f} {bbox_h:.6f}\n")

                # Draw bbox on original image for debugging
                if pixel_bbox is not None:
                    vis_img = img.copy()
                    x, y, box_w, box_h = pixel_bbox
                    cv2.rectangle(vis_img, (x, y), (x + box_w, y + box_h), (0, 255, 0), 2)
                    
                    # Add class label
                    label_text = f"{class_dir.name}"
                    cv2.putText(vis_img, label_text, (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    cv2.imwrite(str(dest_vis_dir / img_path.name), vis_img)

    # Preprocessed images
    yaml_path = f"{output_dir}/data.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"train: /Users/lara.howe/Library/CloudStorage/OneDrive-Accenture/Documents/comp vision/major_project/src/datasets/dataset_yolo_warp_contour/images/train\n")
        f.write(f"val: /Users/lara.howe/Library/CloudStorage/OneDrive-Accenture/Documents/comp vision/major_project/src/datasets/dataset_yolo_warp_contour/images/val\n")
        f.write(f"test: /Users/lara.howe/Library/CloudStorage/OneDrive-Accenture/Documents/comp vision/major_project/src/datasets/dataset_yolo_warp_contour/images/test\n\n")
        f.write(f"nc: {len(class_names)}\n")
        f.write("names: [\n")
        for i, name in enumerate(class_names):
            comma = "," if i < len(class_names) - 1 else ""
            f.write(f"  '{name}'{comma}\n")
        f.write("]\n")