import cv2
import numpy as np
import torch
from ultralytics import YOLO

# =========================================================
# 🔧 MODEL INITIALIZATION
# =========================================================

device = 0 if torch.cuda.is_available() else 'cpu'

# Load models once (global)
contour_model = YOLO("runs_chess/model_warp_contour/weights/best.pt")
colour_model  = YOLO("runs_chess/model_warp_colour/weights/best.pt")

print(f"[INFO] Chess detection models loaded on device: {device}")


# =========================================================
# 🎨 COLOR THRESHOLDING
# =========================================================
def color_threshold_pieces(warped):
    """
    Segment cream/white and black pieces using HSV thresholds
    consistent with training dataset.
    """
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)

    # Cream/white pieces (blue-tinted under LED)
    lower_cream = np.array([90, 50, 50])     # Hue ~90–130
    upper_cream = np.array([130, 255, 255])

    # Dark/black pieces (low brightness)
    lower_dark = np.array([0, 0, 0])
    upper_dark = np.array([180, 255, 60])

    # Create binary masks
    mask_cream = cv2.inRange(hsv, lower_cream, upper_cream)
    mask_black = cv2.inRange(hsv, lower_dark, upper_dark)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_cream = cv2.morphologyEx(mask_cream, cv2.MORPH_OPEN, kernel, iterations=2)
    mask_black = cv2.morphologyEx(mask_black, cv2.MORPH_OPEN, kernel, iterations=2)

    return mask_cream, mask_black


# =========================================================
# 🧱 CONTOUR TO BOUNDING BOXES
# =========================================================
def get_piece_bboxes_from_mask(mask, min_area=300):
    """
    Extract bounding boxes (x1, y1, x2, y2) from binary mask.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        boxes.append([x, y, x + w, y + h])
    return boxes


# =========================================================
# 🧠 COLOR MODEL INFERENCE
# =========================================================
def detect_with_color_model(warped, boxes, expected_color):
    """
    Run color model on ROIs and keep predictions matching expected_color.
    Retries once if predicted color mismatches.
    """
    predictions = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        crop = warped[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        preds = colour_model.predict(source=crop, conf=0.2, imgsz=224, device=device, verbose=False)[0]
        if len(preds.boxes) == 0:
            continue

        cls_id = int(preds.boxes.cls[0].cpu().numpy())
        cls_name = preds.names[cls_id]
        conf = float(preds.boxes.conf[0].cpu().numpy())

        # Retry if mismatch color
        if expected_color not in cls_name:
            preds = colour_model.predict(source=crop, conf=0.2, imgsz=224, device=device, verbose=False)[0]
            if len(preds.boxes) == 0:
                continue
            cls_id = int(preds.boxes.cls[0].cpu().numpy())
            cls_name = preds.names[cls_id]
            conf = float(preds.boxes.conf[0].cpu().numpy())
            if expected_color not in cls_name:
                continue

        predictions.append({
            'box': box,
            'confidence': conf,
            'class_id': cls_id,
            'class_name': cls_name
        })
    return predictions


# =========================================================
# 🔍 VERIFY WITH CONTOUR MODEL
# =========================================================
def verify_with_contour_model(warped, color_predictions):
    """
    Double-check piece type using contour model to ensure classification consistency.
    """
    verified_predictions = []
    for pred in color_predictions:
        x1, y1, x2, y2 = map(int, pred['box'])
        crop = warped[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        contour_preds = contour_model.predict(source=crop, conf=0.2, imgsz=224, device=device, verbose=False)[0]
        if len(contour_preds.boxes) == 0:
            verified_predictions.append(pred)
            continue

        contour_cls_id = int(contour_preds.boxes.cls[0].cpu().numpy())
        contour_cls_name = contour_preds.names[contour_cls_id]

        # Check type consistency (ignore color)
        color_type = pred['class_name'].split('_')[-1]
        contour_type = contour_cls_name.split('_')[-1]
        if color_type != contour_type:
            pred['class_name'] = pred['class_name'].replace(color_type, contour_type)

        verified_predictions.append(pred)

    return verified_predictions


# =========================================================
# ♟️ MASTER DETECTION PIPELINE
# =========================================================
def detect_pieces_colour(warped, visualize=False):
    """
    Full pipeline:
    1. Segment color masks
    2. Extract boxes
    3. Run color model
    4. Verify with contour model
    Returns: list of dicts [{box, confidence, class_id, class_name}]
    """
    if warped is None:
        return []

    # Step 1: color segmentation
    mask_white, mask_black = color_threshold_pieces(warped)

    # Step 2: get bounding boxes
    white_boxes = get_piece_bboxes_from_mask(mask_white)
    black_boxes = get_piece_bboxes_from_mask(mask_black)

    # Step 3: color classification
    white_preds = detect_with_color_model(warped, white_boxes, expected_color='white')
    black_preds = detect_with_color_model(warped, black_boxes, expected_color='black')

    # Step 4: combine and verify
    combined_preds = white_preds + black_preds
    verified_preds = verify_with_contour_model(warped, combined_preds)

    # Optional visualization
    if visualize:
        vis = warped.copy()
        for p in verified_preds:
            x1, y1, x2, y2 = map(int, p['box'])
            label = f"{p['class_name']} ({p['confidence']:.2f})"
            color = (0, 255, 0) if 'white' in p['class_name'] else (0, 0, 255)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.imshow("Detected Pieces", vis)
        cv2.waitKey(1)

    return verified_preds
