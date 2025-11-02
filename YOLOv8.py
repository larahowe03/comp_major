from ultralytics import YOLO
import torch
import shutil
from pathlib import Path

model_name = "yolov8n.pt" 
data_yaml = "dataset_yolo_warp_contour/data.yaml"
epochs = 500
img_size = 426
batch_size = 16
device = 0 if torch.cuda.is_available() else 'cpu'

print(f"🚀 Training YOLOv8 on {device} using {data_yaml}") 

# -------------------------------
# 2️⃣ Load model
# -------------------------------
model = YOLO(model_name)

# -------------------------------
# 3️⃣ Train
# -------------------------------
results = model.train(
    data=data_yaml,      # path to data.yaml
    epochs=epochs,
    imgsz=img_size,
    batch=batch_size,
    device=device,
    name="model_warp_contour",
    project="runs_chess",
    workers=2,
    optimizer='Adam',     # faster convergence for small datasets
    lr0=0.0001,            # learning rate
    patience=10,          # early stopping patience
    verbose=True
)

# -------------------------------
# 4️⃣ Save model
# -------------------------------
# -------------------------------
# 4️⃣ Save model (custom copy)
# -------------------------------
# Get YOLO's best model path
best_model_path = Path("runs_chess/chess_yolov8/weights/best.pt")
# Copy it to a clean filename
save_path = Path("yolov8_chess_best.pt")
if best_model_path.exists():
    shutil.copy(best_model_path, save_path)
    print(f"\n✅ Training complete! Model saved to: {save_path.resolve()}")
else:
    print("⚠️ Could not find YOLO's best weights file. Check your training run directory.")
# -------------------------------
# 5️⃣ (Optional) Evaluate
# -------------------------------
metrics = model.val(data=data_yaml)
print(f"\n📊 Validation metrics: {metrics}")

# -------------------------------
# 6️⃣ (Optional) Inference test
# -------------------------------
# You can test a random image from your test set
test_image = "dataset_yolo/images/test/IMG_1133.jpg"  # example path
print("\n🔍 Running a quick test inference...")
results = model.predict(source=test_image, save=True, conf=0.5)
print("✅ Prediction complete — check the 'runs/detect/predict' folder.")
