import cv2
import numpy as np
import os
from pathlib import Path
import random
from tqdm import tqdm
import shutil

# ----------------------------
# 1️⃣ Config
# ----------------------------
INPUT_IMAGES = "dataset_yolo/images/train"
INPUT_LABELS = "dataset_yolo/labels/train"
OUTPUT_IMAGES = "dataset_yolo/images/train_augmented"
OUTPUT_LABELS = "dataset_yolo/labels/train_augmented"

# Number of augmented versions per image
AUGMENTATIONS_PER_IMAGE = 5

# Create output directories
os.makedirs(OUTPUT_IMAGES, exist_ok=True)
os.makedirs(OUTPUT_LABELS, exist_ok=True)

# ----------------------------
# 2️⃣ FAST Augmentation Functions
# ----------------------------

def random_rotation(image, boxes, angle=None):
    """Rotate image and adjust bounding boxes"""
    if angle is None:
        angle = random.uniform(-30, 30)
    
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    new_boxes = []
    for box in boxes:
        class_id, x_center, y_center, width, height = box
        
        x1 = (x_center - width/2) * w
        y1 = (y_center - height/2) * h
        x2 = (x_center + width/2) * w
        y2 = (y_center + height/2) * h
        
        corners = np.array([[x1, y1, 1], [x2, y1, 1], [x2, y2, 1], [x1, y2, 1]])
        rotated_corners = M.dot(corners.T).T
        
        x_coords = rotated_corners[:, 0]
        y_coords = rotated_corners[:, 1]
        
        new_x1 = max(0, min(x_coords))
        new_y1 = max(0, min(y_coords))
        new_x2 = min(w, max(x_coords))
        new_y2 = min(h, max(y_coords))
        
        new_x_center = ((new_x1 + new_x2) / 2) / w
        new_y_center = ((new_y1 + new_y2) / 2) / h
        new_width = (new_x2 - new_x1) / w
        new_height = (new_y2 - new_y1) / h
        
        if new_width > 0.01 and new_height > 0.01:
            new_boxes.append([class_id, new_x_center, new_y_center, new_width, new_height])
    
    return rotated, new_boxes


def random_stretch(image, boxes):
    """Stretch image horizontally or vertically"""
    h, w = image.shape[:2]
    
    x_stretch = random.uniform(0.7, 1.4)
    y_stretch = random.uniform(0.7, 1.4)
    
    new_w = int(w * x_stretch)
    new_h = int(h * y_stretch)
    
    stretched = cv2.resize(image, (new_w, new_h))
    
    if new_w > w:
        start_x = (new_w - w) // 2
        stretched = stretched[:, start_x:start_x+w]
    elif new_w < w:
        pad_left = (w - new_w) // 2
        pad_right = w - new_w - pad_left
        stretched = cv2.copyMakeBorder(stretched, 0, 0, pad_left, pad_right, cv2.BORDER_REFLECT)
    
    if new_h > h:
        start_y = (new_h - h) // 2
        stretched = stretched[start_y:start_y+h, :]
    elif new_h < h:
        pad_top = (h - new_h) // 2
        pad_bottom = h - new_h - pad_top
        stretched = cv2.copyMakeBorder(stretched, pad_top, pad_bottom, 0, 0, cv2.BORDER_REFLECT)
    
    new_boxes = []
    for box in boxes:
        class_id, x_center, y_center, width, height = box
        
        if new_w > w:
            crop_offset = (new_w - w) / 2 / new_w
            x_center = (x_center * x_stretch - crop_offset) / (1.0)
            width = width * x_stretch
        else:
            pad_left = (w - new_w) / 2
            x_center = (x_center * w * x_stretch + pad_left) / w
            width = width * x_stretch
        
        if new_h > h:
            crop_offset = (new_h - h) / 2 / new_h
            y_center = (y_center * y_stretch - crop_offset) / (1.0)
            height = height * y_stretch
        else:
            pad_top = (h - new_h) / 2
            y_center = (y_center * h * y_stretch + pad_top) / h
            height = height * y_stretch
        
        if 0 < x_center < 1 and 0 < y_center < 1 and width > 0.01 and height > 0.01:
            new_boxes.append([class_id, x_center, y_center, width, height])
    
    return stretched, new_boxes


def random_shear(image, boxes):
    """Apply shear transformation"""
    h, w = image.shape[:2]
    shear_x = random.uniform(-0.2, 0.2)
    shear_y = random.uniform(-0.2, 0.2)
    
    M = np.float32([[1, shear_x, 0], [shear_y, 1, 0]])
    sheared = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    new_boxes = []
    for box in boxes:
        class_id, x_center, y_center, width, height = box
        
        x1 = (x_center - width/2) * w
        y1 = (y_center - height/2) * h
        x2 = (x_center + width/2) * w
        y2 = (y_center + height/2) * h
        
        corners = np.array([[x1, y1, 1], [x2, y1, 1], [x2, y2, 1], [x1, y2, 1]])
        sheared_corners = M.dot(corners.T).T
        
        x_coords = sheared_corners[:, 0]
        y_coords = sheared_corners[:, 1]
        
        new_x1 = max(0, min(x_coords))
        new_y1 = max(0, min(y_coords))
        new_x2 = min(w, max(x_coords))
        new_y2 = min(h, max(y_coords))
        
        new_x_center = ((new_x1 + new_x2) / 2) / w
        new_y_center = ((new_y1 + new_y2) / 2) / h
        new_width = (new_x2 - new_x1) / w
        new_height = (new_y2 - new_y1) / h
        
        if new_width > 0.01 and new_height > 0.01:
            new_boxes.append([class_id, new_x_center, new_y_center, new_width, new_height])
    
    return sheared, new_boxes


def random_perspective_warp(image, boxes):
    """Apply perspective transformation"""
    h, w = image.shape[:2]
    
    pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    pts2 = np.float32([
        [random.randint(0, int(w*0.1)), random.randint(0, int(h*0.1))],
        [random.randint(int(w*0.9), w), random.randint(0, int(h*0.1))],
        [random.randint(0, int(w*0.1)), random.randint(int(h*0.9), h)],
        [random.randint(int(w*0.9), w), random.randint(int(h*0.9), h)]
    ])
    
    M = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    new_boxes = []
    for box in boxes:
        class_id, x_center, y_center, width, height = box
        
        x1 = (x_center - width/2) * w
        y1 = (y_center - height/2) * h
        x2 = (x_center + width/2) * w
        y2 = (y_center + height/2) * h
        
        corners = np.array([[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(corners, M)
        
        x_coords = transformed[:, 0, 0]
        y_coords = transformed[:, 0, 1]
        
        new_x1 = max(0, min(x_coords))
        new_y1 = max(0, min(y_coords))
        new_x2 = min(w, max(x_coords))
        new_y2 = min(h, max(y_coords))
        
        new_x_center = ((new_x1 + new_x2) / 2) / w
        new_y_center = ((new_y1 + new_y2) / 2) / h
        new_width = (new_x2 - new_x1) / w
        new_height = (new_y2 - new_y1) / h
        
        if 0 < new_x_center < 1 and 0 < new_y_center < 1 and new_width > 0.01 and new_height > 0.01:
            new_boxes.append([class_id, new_x_center, new_y_center, new_width, new_height])
    
    return warped, new_boxes


def random_crop(image, boxes, crop_ratio=None):
    """Random crop"""
    if crop_ratio is None:
        crop_ratio = random.uniform(0.7, 0.95)
    
    h, w = image.shape[:2]
    new_h, new_w = int(h * crop_ratio), int(w * crop_ratio)
    
    top = random.randint(0, h - new_h)
    left = random.randint(0, w - new_w)
    
    cropped = image[top:top+new_h, left:left+new_w]
    
    new_boxes = []
    for box in boxes:
        class_id, x_center, y_center, width, height = box
        
        x_center_px = x_center * w
        y_center_px = y_center * h
        width_px = width * w
        height_px = height * h
        
        x1 = x_center_px - width_px/2
        y1 = y_center_px - height_px/2
        x2 = x_center_px + width_px/2
        y2 = y_center_px + height_px/2
        
        x1_crop = max(0, x1 - left)
        y1_crop = max(0, y1 - top)
        x2_crop = min(new_w, x2 - left)
        y2_crop = min(new_h, y2 - top)
        
        if x2_crop > x1_crop and y2_crop > y1_crop:
            new_x_center = ((x1_crop + x2_crop) / 2) / new_w
            new_y_center = ((y1_crop + y2_crop) / 2) / new_h
            new_width = (x2_crop - x1_crop) / new_w
            new_height = (y2_crop - y1_crop) / new_h
            
            new_boxes.append([class_id, new_x_center, new_y_center, new_width, new_height])
    
    return cropped, new_boxes


def random_flip(image, boxes):
    """Horizontal and/or vertical flip"""
    flip_horizontal = random.choice([True, False])
    flip_vertical = random.choice([True, False])
    
    if flip_horizontal:
        image = cv2.flip(image, 1)
    if flip_vertical:
        image = cv2.flip(image, 0)
    
    new_boxes = []
    for box in boxes:
        class_id, x_center, y_center, width, height = box
        
        if flip_horizontal:
            x_center = 1 - x_center
        if flip_vertical:
            y_center = 1 - y_center
        
        new_boxes.append([class_id, x_center, y_center, width, height])
    
    return image, new_boxes


def adjust_brightness_contrast(image):
    """Random brightness and contrast"""
    alpha = random.uniform(0.7, 1.3)
    beta = random.randint(-30, 30)
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def add_noise(image):
    """Add random noise"""
    if random.random() < 0.5:
        sigma = random.uniform(5, 15)
        noise = np.random.normal(0, sigma, image.shape).astype(np.uint8)
        return cv2.add(image, noise)
    return image


def random_blur(image):
    """Apply random blur"""
    if random.random() < 0.5:
        ksize = random.choice([3, 5])
        return cv2.GaussianBlur(image, (ksize, ksize), 0)
    return image


# ----------------------------
# 3️⃣ FAST Augmentation Pipeline (removed slow functions)
# ----------------------------

def augment_image(image, boxes):
    """Apply random combination of FAST augmentations"""
    
    # Geometric transformations only
    if random.random() < 0.6:
        image, boxes = random_rotation(image, boxes)
    
    if random.random() < 0.5:
        image, boxes = random_stretch(image, boxes)
    
    if random.random() < 0.4:
        image, boxes = random_shear(image, boxes)
    
    if random.random() < 0.4:
        image, boxes = random_perspective_warp(image, boxes)
    
    if random.random() < 0.5:
        image, boxes = random_crop(image, boxes)
    
    if random.random() < 0.5:
        image, boxes = random_flip(image, boxes)
    
    return image, boxes

def read_yolo_labels(label_path):
    """Read YOLO format labels"""
    boxes = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    boxes.append([class_id, x_center, y_center, width, height])
    return boxes


def write_yolo_labels(label_path, boxes):
    """Write YOLO format labels"""
    with open(label_path, 'w') as f:
        for box in boxes:
            class_id, x_center, y_center, width, height = box
            f.write(f"{int(class_id)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


# ----------------------------
# 4️⃣ Process All Images
# ----------------------------

image_files = list(Path(INPUT_IMAGES).glob("*.png")) + \
              list(Path(INPUT_IMAGES).glob("*.jpg")) + \
              list(Path(INPUT_IMAGES).glob("*.jpeg"))

print(f"🖼️  Found {len(image_files)} training images")
print(f"🔄 Generating {AUGMENTATIONS_PER_IMAGE} augmentations per image")
print(f"📊 Total output: {len(image_files) * (AUGMENTATIONS_PER_IMAGE + 1)} images\n")

# Copy originals
print("📋 Copying original images...")
for img_path in tqdm(image_files):
    shutil.copy(img_path, os.path.join(OUTPUT_IMAGES, img_path.name))
    
    label_name = img_path.stem + ".txt"
    label_path = os.path.join(INPUT_LABELS, label_name)
    if os.path.exists(label_path):
        shutil.copy(label_path, os.path.join(OUTPUT_LABELS, label_name))

# Generate augmentations
print("\n🎨 Generating FAST augmentations...")
for img_path in tqdm(image_files):
    image = cv2.imread(str(img_path))
    if image is None:
        continue
    
    label_name = img_path.stem + ".txt"
    label_path = os.path.join(INPUT_LABELS, label_name)
    boxes = read_yolo_labels(label_path)
    
    for aug_idx in range(AUGMENTATIONS_PER_IMAGE):
        aug_image, aug_boxes = augment_image(image.copy(), boxes.copy())
        
        if len(aug_boxes) == 0:
            continue
        
        aug_img_name = f"{img_path.stem}_aug{aug_idx}{img_path.suffix}"
        aug_img_path = os.path.join(OUTPUT_IMAGES, aug_img_name)
        cv2.imwrite(aug_img_path, aug_image)
        
        aug_label_name = f"{img_path.stem}_aug{aug_idx}.txt"
        aug_label_path = os.path.join(OUTPUT_LABELS, aug_label_name)
        write_yolo_labels(aug_label_path, aug_boxes)

print("\n✅ Augmentation complete!")
print(f"📁 Augmented images saved to: {OUTPUT_IMAGES}")
print(f"📁 Augmented labels saved to: {OUTPUT_LABELS}")

final_count = len(list(Path(OUTPUT_IMAGES).glob("*")))
print(f"🎯 Total images (original + augmented): {final_count}")