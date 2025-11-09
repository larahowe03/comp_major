import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil
import yaml
from ..training.split_dataset_for_YOLO_contour import preprocess_image

def preprocess_dataset(input_dir, output_dir, copy_labels=True):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
        
    # Check if input data.yaml exists
    input_yaml = input_path / "data.yaml"
    
    if not input_yaml.exists():
        return
    
    # Load yaml paths
    with open(input_yaml, 'r') as f:
        config = yaml.safe_load(f)
    
    # Process each split
    splits = ['train', 'val', 'test']
    
    for split in splits:
        if split not in config:
            continue
        
        print(f"\n{'='*70}")
        print(f"Processing {split.upper()} split")
        print('='*70)
                
        # Get image directory
        img_dir = input_path / "images" / split
        
        if not img_dir.exists():
            continue
                
        # Create output directories
        output_img_dir = output_path / "images" / split
        output_img_dir.mkdir(parents=True, exist_ok=True)
        
        if copy_labels:
            label_dir = input_path / "labels" / split
            output_label_dir = output_path / "labels" / split
            output_label_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all images
        image_files = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.jpeg')) + list(img_dir.glob('*.png')) + list(img_dir.glob('*.bmp'))
        
        print(f"Found {len(image_files)} images")
                
        for img_path in tqdm(image_files, desc=f"Preprocessing {split}"):            
            # Read image
            image = cv2.imread(str(img_path))
                                        
            # Apply same preprocessing as training
            preprocessed = preprocess_image(image)
            output_img_path = output_img_dir / img_path.name
            cv2.imwrite(str(output_img_path), preprocessed)
            
            # Copy corresponding label
            if copy_labels:
                label_path = label_dir / (img_path.stem + '.txt')
                if label_path.exists():
                    output_label_path = output_label_dir / (img_path.stem + '.txt')
                    shutil.copy2(label_path, output_label_path)
        
                    
    # Copy data.yaml
    output_yaml = output_path / "data.yaml"
    
    # Update path in config
    new_config = config.copy()
    new_config['path'] = str(output_path.absolute())
    
    with open(output_yaml, 'w') as f:
        yaml.dump(new_config, f, default_flow_style=False)

if __name__ == "__main__":
    preprocess_dataset(
        input_dir='datasets/kaggle_dataset_warped_remapped',
        output_dir='datasets/kaggle_dataset_warped_remapped_contours',
        copy_labels=True
    )
    