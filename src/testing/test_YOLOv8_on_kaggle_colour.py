from ultralytics import YOLO
import torch
import pandas as pd
import os
import numpy as np
import yaml
from pathlib import Path

def create_temp_yaml(original_yaml, split):
    with open(original_yaml, 'r') as f:
        config = yaml.safe_load(f)
    
    # Modify to point val to the desired split
    temp_config = config.copy()
    
    if split == 'train':
        temp_config['val'] = config.get('train', 'images/train')
    elif split == 'test' and 'test' in config:
        temp_config['val'] = config['test']
    
    # Save temporary yaml
    temp_yaml_path = f"temp_data_{split}.yaml"
    with open(temp_yaml_path, 'w') as f:
        yaml.dump(temp_config, f, default_flow_style=False)
    
    return temp_yaml_path


def evaluate_yolo_all_splits(model_path, data_yaml, output_dir, skip_class_ids=None, imgsz=448, conf=0.1):
    if skip_class_ids is None:
        skip_class_ids = []

    device = 0 if torch.cuda.is_available() else 'cpu'
    model = YOLO(model_path)

    os.makedirs(output_dir, exist_ok=True)

    # Check which splits exist in the data.yaml
    with open(data_yaml, 'r') as f:
        config = yaml.safe_load(f)
    
    available_splits = []  # val is always available
    if 'train' in config:
        available_splits.append('train')
    if 'test' in config:
        available_splits.append('test')

    for split in available_splits:
        # Create temporary yaml for this split if not 'val'
        if split == 'val':
            eval_yaml = data_yaml
        else:
            eval_yaml = create_temp_yaml(data_yaml, split)
        
        # Run validation
        metrics = model.val(
            data=eval_yaml,
            imgsz=imgsz,
            conf=conf,
            device=device,
            verbose=True,
            save_json=False,
            plots=False
        )
        
        # Clean up temp yaml
        if split != 'val' and os.path.exists(eval_yaml):
            os.remove(eval_yaml)
        
        # Extract overall metrics
        overall = pd.DataFrame([{
            'split': split,
            'mAP50': float(metrics.box.map50),
            'mAP50-95': float(metrics.box.map),
            'precision': float(metrics.box.mp),
            'recall': float(metrics.box.mr),
        }])
        overall.to_csv(f"{output_dir}/{split}_overall_metrics.csv", index=False)

        # Extract per-class metrics
        names = metrics.names
        
        # Get all class IDs
        if isinstance(names, dict):
            class_ids = np.array(list(names.keys()))
        else:  # list
            class_ids = np.arange(len(names))
        
        # Get metric arrays and ensure they're numpy arrays
        p = np.array(metrics.box.p) if hasattr(metrics.box, 'p') else np.full(len(class_ids), np.nan)
        r = np.array(metrics.box.r) if hasattr(metrics.box, 'r') else np.full(len(class_ids), np.nan)
        ap50 = np.array(metrics.box.ap50) if hasattr(metrics.box, 'ap50') else np.full(len(class_ids), np.nan)
        ap = np.array(metrics.box.ap) if hasattr(metrics.box, 'ap') else np.full(len(class_ids), np.nan)
        
        # Ensure arrays are the right length
        if len(p) != len(class_ids):
            p = np.full(len(class_ids), np.nan)
        if len(r) != len(class_ids):
            r = np.full(len(class_ids), np.nan)
        if len(ap50) != len(class_ids):
            ap50 = np.full(len(class_ids), np.nan)
        if len(ap) != len(class_ids):
            ap = np.full(len(class_ids), np.nan)
        
        # Create full per-class dataframe
        per_class_full = pd.DataFrame({
            'class_id': class_ids,
            'class_name': [names[i] for i in class_ids],
            'precision': p,
            'recall': r,
            'mAP50': ap50,
            'mAP50-95': ap
        })
        
        # Save full results
        per_class_full.to_csv(f"{output_dir}/{split}_per_class_metrics_full.csv", index=False)
        
        # Create filtered version (excluding skipped classes)
        if skip_class_ids:
            keep_mask = ~np.isin(class_ids, skip_class_ids)
            per_class_filtered = per_class_full[keep_mask].reset_index(drop=True)
            per_class_filtered.to_csv(f"{output_dir}/{split}_per_class_metrics.csv", index=False)
        else:
            per_class_full.to_csv(f"{output_dir}/{split}_per_class_metrics.csv", index=False)            

if __name__ == "__main__":    
    skip_classes = []

    evaluate_yolo_all_splits(
        model_path="models/YOLOv8/model_warp_contour/weights/best.pt",
        data_yaml="datasets/kaggle_dataset_warped_remapped_contours/data.yaml",
        output_dir="results/contour_model_metrics_on_kaggle",
        skip_class_ids=skip_classes,
        imgsz=448,
        conf=0.01
    )
    
    evaluate_yolo_all_splits(
        model_path=f"models/YOLOv8/model_warp_colour/weights/best.pt",
        data_yaml=f"datasets/kaggle_dataset_warped_remapped/data.yaml",
        output_dir=f"results/colour_model_metrics_on_kaggle",
        skip_class_ids=skip_classes,
        imgsz=448,
        conf=0.01  # LOWERED from 0.1 to catch low-confidence predictions
    )

