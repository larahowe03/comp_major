from ultralytics import YOLO
import torch
import pandas as pd
import os
import numpy as np
import yaml

def evaluate_yolo_test(model_path, data_yaml, output_dir, skip_class_ids=None, imgsz=448, conf=0.01):
    if skip_class_ids is None:
        skip_class_ids = []

    device = 0 if torch.cuda.is_available() else 'cpu'
    model = YOLO(model_path)

    os.makedirs(output_dir, exist_ok=True)

    # Load config
    with open(data_yaml, 'r') as f:
        config = yaml.safe_load(f)
            
    # Create temporary yaml pointing val to test
    temp_config = config.copy()
    temp_config['val'] = config['test']
    
    temp_yaml_path = "temp_data_test.yaml"
    with open(temp_yaml_path, 'w') as f:
        yaml.dump(temp_config, f, default_flow_style=False)
    
    # Run validation on test set
    metrics = model.val(
        data=temp_yaml_path,
        imgsz=imgsz,
        conf=conf,
        device=device,
        verbose=True,
        save_json=False,
        plots=False
    )
    
    # Extract overall metrics
    overall = pd.DataFrame([{
        'split': 'test',
        'mAP50': float(metrics.box.map50),
        'mAP50-95': float(metrics.box.map),
        'precision': float(metrics.box.mp),
        'recall': float(metrics.box.mr),
    }])
    overall.to_csv(f"{output_dir}/test_overall_metrics.csv", index=False)
    
    # Extract per-class metrics
    names = metrics.names
    
    # Get all class IDs
    if isinstance(names, dict):
        class_ids = np.array(list(names.keys()))
    else:  # list
        class_ids = np.arange(len(names))
    
    # Get metric arrays
    p = np.array(metrics.box.p) if hasattr(metrics.box, 'p') and len(metrics.box.p) > 0 else np.full(len(class_ids), np.nan)
    r = np.array(metrics.box.r) if hasattr(metrics.box, 'r') and len(metrics.box.r) > 0 else np.full(len(class_ids), np.nan)
    ap50 = np.array(metrics.box.ap50) if hasattr(metrics.box, 'ap50') and len(metrics.box.ap50) > 0 else np.full(len(class_ids), np.nan)
    ap = np.array(metrics.box.ap) if hasattr(metrics.box, 'ap') and len(metrics.box.ap) > 0 else np.full(len(class_ids), np.nan)
    
    # Ensure arrays are the right length
    if len(p) != len(class_ids):
        p = np.full(len(class_ids), np.nan)
    if len(r) != len(class_ids):
        r = np.full(len(class_ids), np.nan)
    if len(ap50) != len(class_ids):
        ap50 = np.full(len(class_ids), np.nan)
    if len(ap) != len(class_ids):
        ap = np.full(len(class_ids), np.nan)
    
    # Create per-class dataframe
    per_class_full = pd.DataFrame({
        'class_id': class_ids,
        'class_name': [names[i] for i in class_ids],
        'precision': p,
        'recall': r,
        'mAP50': ap50,
        'mAP50-95': ap
    })
    
    # Save full results
    per_class_full.to_csv(f"{output_dir}/test_per_class_metrics_full.csv", index=False)
    
    # Create filtered version if needed
    if skip_class_ids:
        keep_mask = ~np.isin(class_ids, skip_class_ids)
        per_class_filtered = per_class_full[keep_mask].reset_index(drop=True)
        per_class_filtered.to_csv(f"{output_dir}/test_per_class_metrics.csv", index=False)
    else:
        per_class_full.to_csv(f"{output_dir}/test_per_class_metrics.csv", index=False)
        
    # Clean up temp yaml
    if os.path.exists(temp_yaml_path):
        os.remove(temp_yaml_path)


if __name__ == "__main__":    
    skip_classes = []
    
    evaluate_yolo_test(
        model_path="models/YOLOv8/model_warp_contour/weights/best.pt",
        data_yaml="datasets/kaggle_dataset_warped_remapped_contours/data.yaml",
        output_dir="results/contour_model_metrics_on_kaggle",
        skip_class_ids=skip_classes,
        imgsz=448,
        conf=0.01
    )