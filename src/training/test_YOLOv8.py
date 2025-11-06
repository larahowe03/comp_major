from ultralytics import YOLO
import torch
from pathlib import Path
import os
import pandas as pd

def evaluate_yolo_all_splits():
    """Evaluate YOLO model on train, val, and test sets"""
    
    # ----------------------------
    # 1️⃣ Config
    # ----------------------------
    model_path = "runs_chess/model_warp_contour/weights/best.pt"
    data_yaml = "dataset_yolo_warp_contour/data.yaml"  # Path to your data.yaml
    
    device = 0 if torch.cuda.is_available() else 'cpu'
    model = YOLO(model_path)
    
    print(f"🔍 Using device: {device}\n")
    
    # ----------------------------
    # 2️⃣ Evaluate on each split
    # ----------------------------
    splits = ['train', 'val', 'test']
    results_dict = {}
    
    for split in splits:
        print(f"{'='*60}")
        print(f"📊 Evaluating on {split.upper()} set")
        print(f"{'='*60}\n")
        
        # Run validation (works for any split)
        metrics = model.val(
            data=data_yaml,
            split=split,  # 'train', 'val', or 'test'
            imgsz=448,
            conf=0.1,
            device=device,
            verbose=True
        )
        
        # Extract metrics
        results_dict[split] = {
            'mAP50': float(metrics.box.map50),
            'mAP50-95': float(metrics.box.map),
            'precision': float(metrics.box.mp),
            'recall': float(metrics.box.mr),
        }
        
        print(f"\n{split.upper()} Results:")
        print(f"  mAP50:    {results_dict[split]['mAP50']:.4f}")
        print(f"  mAP50-95: {results_dict[split]['mAP50-95']:.4f}")
        print(f"  Precision: {results_dict[split]['precision']:.4f}")
        print(f"  Recall:    {results_dict[split]['recall']:.4f}")
        print()
    
    # ----------------------------
    # 3️⃣ Create summary DataFrame
    # ----------------------------
    df = pd.DataFrame(results_dict).T
    df.index.name = 'split'
    
    print(f"{'='*60}")
    print("📈 SUMMARY - All Splits")
    print(f"{'='*60}\n")
    print(df.to_string())
    print()
    
    # ----------------------------
    # 4️⃣ Save to CSV
    # ----------------------------
    output_file = "model_metrics_all_splits.csv"
    df.to_csv(output_file)
    print(f"✅ Metrics saved to: {output_file}\n")
    
    return df


if __name__ == "__main__":
    metrics_df = evaluate_yolo_all_splits()