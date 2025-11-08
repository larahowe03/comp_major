from ultralytics import YOLO
import torch
import pandas as pd
import os

def evaluate_yolo_all_splits(model_path, data_yaml, output_dir):
    device = 0 if torch.cuda.is_available() else 'cpu'
    model = YOLO(model_path)

    os.makedirs(output_dir, exist_ok=True)

    splits = ['train', 'val', 'test']
    
    for split in splits:
        metrics = model.val(
            data=data_yaml,
            split=split,
            imgsz=448,
            conf=0.1,
            device=device,
            verbose=True
        )
        
        # overall
        overall = pd.DataFrame([{
            'split': split,
            'mAP50': float(metrics.box.map50),
            'mAP50-95': float(metrics.box.map),
            'precision': float(metrics.box.mp),
            'recall': float(metrics.box.mr),
        }])
        overall.to_csv(f"{output_dir}/{split}_overall_metrics.csv", index=False)

        # per-class metric
        names = metrics.names
        per_class = pd.DataFrame({
            'class_id': list(names.keys()),
            'class_name': list(names.values()),
            'precision': metrics.box.p.tolist(),
            'recall': metrics.box.r.tolist(),
            'mAP50': metrics.box.ap50.tolist(),
            'mAP50-95': metrics.box.ap.tolist()
        })

        per_class.to_csv(f"{output_dir}/{split}_per_class_metrics.csv", index=False)

for model_type in ["colour", "contour"]:
    evaluate_yolo_all_splits(
        model_path=f"models/YOLOv8/model_warp_{model_type}/weights/best.pt",
        data_yaml=f"datasets/dataset_yolo_warp_{model_type}/data.yaml",
        output_dir=f"results/{model_type}_model_metrics"
    )
