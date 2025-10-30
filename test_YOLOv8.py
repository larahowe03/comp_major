from ultralytics import YOLO
import torch
import cv2
import os
import numpy as np
from warp_board import process_chess_image

# ----------------------------
# 1️⃣ Config
# ----------------------------
model_path = "runs_chess/chess_yolov83/weights/best.pt"
test_image = "img.png"
save_dir = "runs_chess/test_results"
cell_pred_dir = os.path.join(save_dir, "cell_predictions")
os.makedirs(cell_pred_dir, exist_ok=True)

# ----------------------------
# 2️⃣ Load YOLO
# ----------------------------
device = 0 if torch.cuda.is_available() else 'cpu'
model = YOLO(model_path)
print(f"🔍 Using device: {device}")

# ----------------------------
# 3️⃣ Warp board & split into cells
# ----------------------------
warp = process_chess_image(test_image)
h, w = warp.shape[:2]
rows, cols = 8, 8
cell_h, cell_w = h // rows, w // cols

predicted_board = []

# ----------------------------
# 4️⃣ Loop through cells
# ----------------------------
for i in range(rows):
    row_preds = []
    for j in range(cols):
        y1, y2 = i * cell_h, (i + 1) * cell_h
        x1, x2 = j * cell_w, (j + 1) * cell_w

        # Optional small padding around cell
        pad = 10
        y1p, y2p = max(0, y1 - pad), min(h, y2 + pad)
        x1p, x2p = max(0, x1 - pad), min(w, x2 + pad)
        cell = warp[y1p:y2p, x1p:x2p]

        # Skip empty cells quickly
        if cell.mean() < 25:
            pred_class = "empty"
        else:
            # Run YOLO prediction on the cropped cell
            results = model.predict(
                source=cell,
                conf=0.4,
                imgsz=256,
                device=device,
                save=True,                   # save the annotated image
                project=cell_pred_dir,        # output dir
                name=f"r{i+1}_c{j+1}",        # subfolder name
                exist_ok=True,
                verbose=False
            )

            preds = results[0]
            if len(preds.boxes) == 0:
                pred_class = "empty"
            else:
                cls_id = int(preds.boxes[0].cls)
                pred_class = preds.names[cls_id]

        row_preds.append(pred_class)

        print(f"Cell ({i+1},{j+1}) → {pred_class}")

    predicted_board.append(row_preds)

# ----------------------------
# 5️⃣ Done
# ----------------------------
print("\n✅ Saved all annotated YOLO cell predictions!")
print(f"📁 Folder: {cell_pred_dir}")
