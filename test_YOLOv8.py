from ultralytics import YOLO
import torch
from pathlib import Path
import os

def test_yolo_all_simple():
    """Test YOLO on all test images - simple version"""
    
    # ----------------------------
    # 1️⃣ Config
    # ----------------------------
    model_path = "runs_chess/chess_yolov816/weights/best.pt"
    test_images_dir = Path("dataset_yolo/images/test")  # Original images
    # test_images_dir = Path("dataset_yolo/images_preprocessed/test")  # Or preprocessed
    save_dir = "runs_chess/test_results"
    os.makedirs(save_dir, exist_ok=True)

    # ----------------------------
    # 2️⃣ Load YOLO
    # ----------------------------
    device = 0 if torch.cuda.is_available() else 'cpu'
    model = YOLO(model_path)
    print(f"🔍 Using device: {device}")

    # ----------------------------
    # 3️⃣ Get all test images
    # ----------------------------
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    test_images = []
    for ext in image_extensions:
        test_images.extend(test_images_dir.glob(ext))
    
    test_images = sorted(test_images)
    print(f"\n📸 Found {len(test_images)} test images\n")

    # ----------------------------
    # 4️⃣ Run predictions on all images
    # ----------------------------
    results = model.predict(
        source=str(test_images_dir),  # Just point to the folder
        conf=0.1,                     # Confidence threshold
        imgsz=448,                     # Image size (match training)
        device=device,
        save=True,                     # Save annotated images
        project=save_dir,              # Save location
        name="predictions",            # Subfolder name
        exist_ok=True,                 # Overwrite if exists
        save_txt=True,                 # Save labels as txt files
        save_conf=True,                # Save confidence scores
        verbose=True
    )

    # ----------------------------
    # 5️⃣ Print summary for each image
    # ----------------------------
    print(f"\n{'='*60}")
    print("📊 PREDICTION SUMMARY")
    print(f"{'='*60}\n")
    
    for idx, result in enumerate(results):
        img_path = result.path
        img_name = Path(img_path).name
        num_detections = len(result.boxes)
        
        print(f"{idx+1}. {img_name}")
        print(f"   Detections: {num_detections}")
        
        if num_detections > 0:
            for box in result.boxes:
                cls_id = int(box.cls)
                conf = float(box.conf)
                class_name = result.names[cls_id]
                print(f"   - {class_name}: {conf:.2f}")
        else:
            print(f"   - No detections")
        print()

    # ----------------------------
    # 6️⃣ Summary
    # ----------------------------
    print(f"{'='*60}")
    print("✅ TESTING COMPLETE")
    print(f"{'='*60}")
    print(f"Total images processed: {len(test_images)}")
    print(f"📁 Results saved to: {save_dir}/predictions/")
    print(f"📁 Annotated images: {save_dir}/predictions/")
    print(f"📁 Label files: {save_dir}/predictions/labels/")
    

if __name__ == "__main__":
    test_yolo_all_simple()