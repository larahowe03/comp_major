from ultralytics import YOLO
import torch
import cv2
import os
import numpy as np
from warp_board import process_chess_image

def test_yolo(test_image):

    # ----------------------------
    # 1️⃣ Config
    # ----------------------------
    model_path = "runs_chess/chess_yolov83/weights/best.pt"
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
    
    # Store all annotated cells for stitching
    annotated_cells = [[None for _ in range(cols)] for _ in range(rows)]

    # ----------------------------
    # 4️⃣ Loop through cells
    # ----------------------------
    for i in range(rows):
        row_preds = []
        for j in range(cols):
            y1, y2 = i * cell_h, (i + 1) * cell_h
            x1, x2 = j * cell_w, (j + 1) * cell_w

            # Extract exact cell (no padding for stitching)
            cell = warp[y1:y2, x1:x2].copy()

            # Skip empty cells quickly
            if cell.mean() < 25:
                pred_class = "empty"
                annotated_cells[i][j] = cell
            else:
                # Run YOLO prediction on the cropped cell
                results = model.predict(
                    source=cell,
                    conf=0.1,
                    imgsz=256,
                    device=device,
                    save=False,  # We'll handle saving ourselves
                    verbose=False
                )

                preds = results[0]
                if len(preds.boxes) == 0:
                    pred_class = "empty"
                    annotated_cells[i][j] = cell
                else:
                    cls_id = int(preds.boxes[0].cls)
                    pred_class = preds.names[cls_id]
                    
                    # Get the annotated image from YOLO results
                    annotated_cell = preds.plot()
                    annotated_cells[i][j] = annotated_cell

            row_preds.append(pred_class)
            # print(f"Cell ({i+1},{j+1}) → {pred_class}")

        predicted_board.append(row_preds)

    # ----------------------------
    # 5️⃣ Stitch all cells together
    # ----------------------------
    print("\n🧩 Stitching cells together...")
    
    # Stitch rows first
    stitched_rows = []
    for i in range(rows):
        row_img = np.hstack(annotated_cells[i])
        stitched_rows.append(row_img)
    
    # Stitch all rows together
    stitched_img = np.vstack(stitched_rows)
    
    # Save the complete stitched image
    stitched_path = os.path.join(save_dir, "stitched_board_predictions.jpg")
    cv2.imwrite(stitched_path, stitched_img)
    
    print(f"✅ Saved stitched board image!")
    print(f"📁 File: {stitched_path}")

    return stitched_img