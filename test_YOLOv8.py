from ultralytics import YOLO
import torch
import cv2
import os

# ----------------------------
# 1️⃣ Config
# ----------------------------
model_path = "runs_chess/chess_yolov83/weights/best.pt"  # your trained model
test_image = "dataset_yolo/images/test/IMG_1179.png"    # image to test
save_dir = "runs_chess/test_results"                    # where to save
os.makedirs(save_dir, exist_ok=True)

# ----------------------------
# 2️⃣ Load model
# ----------------------------
device = 0 if torch.cuda.is_available() else 'cpu'
print(f"🔍 Using device: {device}")
model = YOLO(model_path)

# ----------------------------
# 3️⃣ Run inference
# ----------------------------
results = model.predict(
    source=test_image,     # path or directory
    conf=0.5,              # confidence threshold (adjust 0.3–0.7)
    imgsz=640,             # image size
    device=device,
    save=True,             # save image with boxes
    project=save_dir,      # output directory
    name="predictions"     # folder name
)

# ----------------------------
# 4️⃣ Print results
# ----------------------------
print("\n✅ Inference complete!")
print(f"📸 Results saved to: {results[0].save_dir}")

# Optional: Show image using OpenCV
img_with_boxes = cv2.imread(str(results[0].save_dir / os.path.basename(test_image)))
cv2.imshow("YOLOv8 Prediction", img_with_boxes)
cv2.waitKey(0)
cv2.destroyAllWindows()