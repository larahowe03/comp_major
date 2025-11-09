import yaml
from pathlib import Path
import cv2
import numpy as np
from ..main.warp_board import detect_board

def load_yolo_config(data_yaml_path):
    with open(data_yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def parse_yolo_annotation(label_path):
    bboxes = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                bboxes.append((class_id, x_center, y_center, width, height))
    return bboxes


def transform_bbox(bbox, M, orig_img_width, orig_img_height, new_img_width, new_img_height):
    class_id, x_center_norm, y_center_norm, width_norm, height_norm = bbox
    
    # Convert normalized to pixel coordinates
    x_center = x_center_norm * orig_img_width
    y_center = y_center_norm * orig_img_height
    width = width_norm * orig_img_width
    height = height_norm * orig_img_height
    
    # Get bbox corners in original image
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2
    
    # Define 4 corners of the bbox
    corners = np.array([
        [x1, y1],
        [x2, y1],
        [x2, y2],
        [x1, y2]
    ], dtype=np.float32)
    
    # Transform corners using perspective transform
    corners_homogeneous = np.concatenate([corners, np.ones((4, 1))], axis=1)
    transformed_corners = (M @ corners_homogeneous.T).T
    
    # Convert from homogeneous coordinates
    transformed_corners = transformed_corners[:, :2] / transformed_corners[:, 2:3]
    
    # Get bounding box of transformed corners
    x_min = np.min(transformed_corners[:, 0])
    y_min = np.min(transformed_corners[:, 1])
    x_max = np.max(transformed_corners[:, 0])
    y_max = np.max(transformed_corners[:, 1])
    
    # Clip to image boundaries
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(new_img_width, x_max)
    y_max = min(new_img_height, y_max)
    
    # Calculate new center and dimensions
    new_width = x_max - x_min
    new_height = y_max - y_min
    
    # Check if bbox is valid (has positive area)
    if new_width <= 0 or new_height <= 0:
        return None
    
    new_x_center = (x_min + x_max) / 2
    new_y_center = (y_min + y_max) / 2
    
    # Normalize to [0, 1]
    new_x_center_norm = new_x_center / new_img_width
    new_y_center_norm = new_y_center / new_img_height
    new_width_norm = new_width / new_img_width
    new_height_norm = new_height / new_img_height
    
    # Clip to valid range
    new_x_center_norm = np.clip(new_x_center_norm, 0, 1)
    new_y_center_norm = np.clip(new_y_center_norm, 0, 1)
    new_width_norm = np.clip(new_width_norm, 0, 1)
    new_height_norm = np.clip(new_height_norm, 0, 1)
    
    return (class_id, new_x_center_norm, new_y_center_norm, new_width_norm, new_height_norm)


def save_yolo_annotation(label_path, bboxes):
    with open(label_path, 'w') as f:
        for bbox in bboxes:
            class_id, x_center, y_center, width, height = bbox
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


def warp_yolo_dataset(data_yaml_path, output_dir, board_size=800, margin=60):
    # Load YOLO configuration
    config = load_yolo_config(data_yaml_path)
    
    # Get class names
    class_names = config['names']
    
    # Get dataset paths
    data_yaml_dir = Path(data_yaml_path).parent
    
    # Process train, val, and test splits
    splits = ['train', 'val']
    if 'test' in config:
        splits.append('test')
            
    for split in splits:
        # Get image directory path
        if split in config:
            img_dir = data_yaml_dir / config[split]
        else:
            img_dir = data_yaml_dir / 'images' / split
        
        # Get labels directory path
        label_dir = Path(str(img_dir).replace('images', 'labels'))
                
        # Create output directories
        output_img_dir = Path(output_dir) / 'images' / split
        output_label_dir = Path(output_dir) / 'labels' / split
        output_img_dir.mkdir(parents=True, exist_ok=True)
        output_label_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each image
        image_files = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.jpeg')) + \
                      list(img_dir.glob('*.png')) + list(img_dir.glob('*.bmp'))
        
        for img_path in image_files:
            
            # Read image
            image = cv2.imread(str(img_path))
            
            orig_h, orig_w = image.shape[:2]
            
            # Get corresponding label file
            label_path = label_dir / (img_path.stem + '.txt')
        
            bboxes = parse_yolo_annotation(label_path)
            
            # Detect board and warp
            warp_margined, warp_unmargined, M_margined, pts_src = detect_board(
                image, board_size=board_size, margin=margin
            )
            
            # Get the transformation matrix for unmargined (800x800) version
            pts_dst_unmargined = np.float32([[0, 0], [board_size, 0], [board_size, board_size], [0, board_size]])
            M_unmargined = cv2.getPerspectiveTransform(pts_src, pts_dst_unmargined)
            
            # Transform all bboxes using the 800x800 transformation
            transformed_bboxes = []
            for bbox in bboxes:
                transformed_bbox = transform_bbox(
                    bbox, M_unmargined, orig_w, orig_h,
                    board_size, board_size  # Use board_size (800x800) not board_size_with_margin
                )
                if transformed_bbox is not None:
                    transformed_bboxes.append(transformed_bbox)

            output_img_path = output_img_dir / img_path.name
            cv2.imwrite(str(output_img_path), cv2.cvtColor(warp_unmargined, cv2.COLOR_RGB2BGR))
            
            # Save transformed lables
            output_label_path = output_label_dir / (img_path.stem + '.txt')
            save_yolo_annotation(output_label_path, transformed_bboxes)
                
    # Create new data.yaml for warped dataset
    new_config = {
        'path': str(Path(output_dir).absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'names': class_names
    }
    
    if 'test' in config:
        new_config['test'] = 'images/test'
    
    output_yaml_path = Path(output_dir) / 'data.yaml'
    with open(output_yaml_path, 'w') as f:
        yaml.dump(new_config, f, default_flow_style=False)
    
if __name__ == "__main__":
    # Configuration
    data_yaml_path = "datasets/kaggle_dataset/data.yaml"
    output_dir = "datasets/kaggle_dataset_warped"
    board_size = 800
    margin = 60
    
    warp_yolo_dataset(data_yaml_path, output_dir, board_size=board_size, margin=margin)