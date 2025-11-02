import cv2
import pickle
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO

from warp_board import process_chess_image
from split_dataset_for_YOLO_contour import preprocess_image


# Model paths
contour_model_path = "runs_chess/model_warp_contour/weights/best.pt"
colour_model_path = "runs_chess/model_warp_colour/weights/best.pt"

# Load models
device = 0 if torch.cuda.is_available() else 'cpu'
contour_model = YOLO(contour_model_path)
colour_model = YOLO(colour_model_path)

print(f"Using device: {device}")


def contour_model_prediction(img):
    result = contour_model.predict(
        source=img,
        conf=0.1,
        imgsz=426,
        device=device,
        verbose=False
    )
    return result[0]


def colour_model_prediction(img):
    result = colour_model.predict(
        source=img,
        conf=0.1,
        imgsz=426,
        device=device,
        verbose=False
    )
    return result[0]


def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two boxes.
    Boxes format: [x1, y1, x2, y2]
    """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    inter_area = inter_width * inter_height
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0
    
    iou = inter_area / union_area
    return iou


def get_piece_type(class_name):
    """
    Extract piece type from class name.
    'white_queen' -> 'queen'
    'queen' -> 'queen'
    """
    if isinstance(class_name, str):
        if '_' in class_name:
            return class_name.split('_')[1]
        return class_name
    return class_name


def classes_match(cls1_id, cls2_id, model1_names, model2_names):
    """
    Check if two class IDs represent the same piece type.
    Handles mismatch between 'queen' and 'white_queen'/'black_queen'.
    """
    cls1_name = model1_names[int(cls1_id)]
    cls2_name = model2_names[int(cls2_id)]
    
    piece1 = get_piece_type(cls1_name)
    piece2 = get_piece_type(cls2_name)
    
    return piece1 == piece2


def ensemble_predictions(contour_result, colour_result, iou_threshold=0.5, conf_weight_contour=0.5):
    """
    Ensemble predictions from contour and color models.
    Uses weighted box fusion to combine overlapping detections.
    Handles class name mismatch: contour uses 'queen', color uses 'white_queen'.
    
    Args:
        contour_result: YOLO result from contour model (classes: queen, king, etc.)
        colour_result: YOLO result from color model (classes: white_queen, black_king, etc.)
        iou_threshold: IoU threshold for considering boxes as duplicates
        conf_weight_contour: Weight for contour model confidence (0-1)
    
    Returns:
        Combined predictions with color model's class names (white_queen, etc.)
    """
    # Extract boxes and confidences
    contour_boxes = contour_result.boxes
    colour_boxes = colour_result.boxes
    
    # If one model has no detections, return the other
    if len(contour_boxes) == 0:
        return colour_result
    if len(colour_boxes) == 0:
        return contour_result
    
    # Get class names from both models
    contour_names = contour_result.names
    colour_names = colour_result.names
    
    # Get box coordinates, confidences, and class IDs
    contour_xyxy = contour_boxes.xyxy.cpu().numpy()
    contour_conf = contour_boxes.conf.cpu().numpy()
    contour_cls = contour_boxes.cls.cpu().numpy()
    
    colour_xyxy = colour_boxes.xyxy.cpu().numpy()
    colour_conf = colour_boxes.conf.cpu().numpy()
    colour_cls = colour_boxes.cls.cpu().numpy()
    
    # Build list of all detections with metadata
    all_detections = []
    
    # Add color boxes (these have full class names like 'white_queen')
    for i in range(len(colour_boxes)):
        all_detections.append({
            'box': colour_xyxy[i],
            'conf': colour_conf[i],
            'cls_id': colour_cls[i],
            'cls_name': colour_names[int(colour_cls[i])],
            'piece_type': get_piece_type(colour_names[int(colour_cls[i])]),
            'source': 'colour',
            'weight': 1 - conf_weight_contour
        })
    
    # Add contour boxes (these have piece type only like 'queen')
    for i in range(len(contour_boxes)):
        all_detections.append({
            'box': contour_xyxy[i],
            'conf': contour_conf[i],
            'cls_id': contour_cls[i],
            'cls_name': contour_names[int(contour_cls[i])],
            'piece_type': get_piece_type(contour_names[int(contour_cls[i])]),
            'source': 'contour',
            'weight': conf_weight_contour
        })
    
    # Perform ensemble using weighted average for overlapping boxes
    final_boxes = []
    final_confs = []
    final_cls = []
    used_indices = set()
    
    for i in range(len(all_detections)):
        if i in used_indices:
            continue
        
        current = all_detections[i]
        
        # Find overlapping boxes of the same piece type
        overlapping = [current]
        overlapping_indices = [i]
        
        for j in range(i + 1, len(all_detections)):
            if j in used_indices:
                continue
            
            other = all_detections[j]
            
            # Check if same piece type (handles 'queen' == 'white_queen')
            if current['piece_type'] != other['piece_type']:
                continue
            
            # Calculate IoU
            iou = calculate_iou(current['box'], other['box'])
            
            if iou > iou_threshold:
                overlapping.append(other)
                overlapping_indices.append(j)
                used_indices.add(j)
        
        used_indices.add(i)
        
        # Combine overlapping detections
        if len(overlapping) > 1:
            # Weighted average of boxes
            boxes = np.array([d['box'] for d in overlapping])
            weights = np.array([d['weight'] for d in overlapping])
            weights = weights / weights.sum()  # Normalize
            avg_box = np.average(boxes, axis=0, weights=weights)
            
            # Weighted average of confidences
            confs = np.array([d['conf'] for d in overlapping])
            avg_conf = np.average(confs, weights=weights)
            # Boost confidence when both models agree
            avg_conf = min(1.0, avg_conf * 1.2)
            
            # Prefer color model's class (has color info)
            color_detections = [d for d in overlapping if d['source'] == 'colour']
            if color_detections:
                # Use the color model's class ID (with highest confidence)
                best_color = max(color_detections, key=lambda x: x['conf'])
                final_cls_id = best_color['cls_id']
            else:
                # Fall back to contour class
                final_cls_id = overlapping[0]['cls_id']
            
            final_boxes.append(avg_box)
            final_confs.append(avg_conf)
            final_cls.append(final_cls_id)
        else:
            # No overlap, keep original
            final_boxes.append(current['box'])
            final_confs.append(current['conf'])
            final_cls.append(current['cls_id'])
    
    # Create a new result object with ensembled predictions
    if len(final_boxes) == 0:
        return colour_result
    
    # Convert to tensors
    final_boxes_tensor = torch.tensor(np.array(final_boxes), device=colour_boxes.xyxy.device, dtype=torch.float32)
    final_confs_tensor = torch.tensor(np.array(final_confs), device=colour_boxes.conf.device, dtype=torch.float32)
    final_cls_tensor = torch.tensor(np.array(final_cls), device=colour_boxes.cls.device, dtype=torch.float32)
    
    # Create new boxes with xywh format (YOLO expects this)
    # Convert xyxy to xywh
    x1y1 = final_boxes_tensor[:, :2]
    x2y2 = final_boxes_tensor[:, 2:]
    wh = x2y2 - x1y1
    xywh = torch.cat([x1y1, wh], dim=1)
    
    # Stack all data: [xyxy (4), conf (1), cls (1)] = 6 columns
    boxes_data = torch.cat([
        final_boxes_tensor,  # xyxy
        final_confs_tensor.unsqueeze(1),  # conf
        final_cls_tensor.unsqueeze(1)  # cls
    ], dim=1)
    
    # Create new Boxes object
    from ultralytics.engine.results import Boxes
    new_boxes = Boxes(boxes_data, colour_result.orig_shape)
    
    # Clone the result and replace boxes
    import copy
    ensemble_result = copy.deepcopy(colour_result)
    ensemble_result.boxes = new_boxes
    
    return ensemble_result


def ensemble_model(contoured_warp, coloured_warp):
    """
    Run both models and ensemble their predictions.
    """
    contour_predictions = contour_model_prediction(contoured_warp)
    colour_predictions = colour_model_prediction(coloured_warp)
    
    # Combine predictions using ensemble
    ensemble_result = ensemble_predictions(
        contour_predictions, 
        colour_predictions,
        iou_threshold=0.5,
        conf_weight_contour=0.5  # Equal weight to both models
    )
    
    return ensemble_result, contour_predictions, colour_predictions


def undistort(img, K, d):
    return cv2.undistort(img, K, d, None, K)


def visualize_predictions(img, result, title="Predictions"):
    """
    Visualize YOLO predictions on image.
    """
    annotated_img = result.plot()
    return annotated_img


if __name__ == "__main__":
    # Phone IP and port from DroidCam app
    phone_ip = "10.19.204.143"
    port = "4747"

    # DroidCam streaming URLs
    urls = [
        f"http://{phone_ip}:{port}/video",
        f"http://{phone_ip}:{port}/mjpegfeed",
    ]

    cap = None
    for url in urls:
        print(f"Trying: {url}")
        cap = cv2.VideoCapture(url)
        if cap.isOpened():
            print(f"Connected successfully to {url}")
            break
        cap.release()

    if not cap or not cap.isOpened():
        print("Could not connect. Check:")
        print("1. Phone and laptop on same WiFi")
        print("2. IP address is correct")
        print("3. DroidCam app is running")
        exit()

    # Load calibration data
    with open("calibration_coefficients.pkl", "rb") as f:
        data = pickle.load(f)

    K = data['camera_matrix']
    d = data['distortion_coeffs']

    print("Starting detection loop. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Connection lost")
            break

        # Undistort frame
        undistorted = undistort(frame, K, d)

        # Warp board
        warped, contoured_img = process_chess_image(undistorted)

        # Display warped board
        if warped is not None:
            cv2.imshow('Warped Board', warped)
            
            # Prepare images for predictions
            contoured_warp = preprocess_image(warped)
            coloured_warp = warped

            # Get ensemble predictions
            ensemble_result, contour_predictions, colour_predictions = ensemble_model(contoured_warp, coloured_warp)

            # Visualize all three
            ensemble_viz = visualize_predictions(coloured_warp, ensemble_result, "Ensemble")
            contour_viz = visualize_predictions(contoured_warp, contour_predictions, "Contour")
            colour_viz = visualize_predictions(coloured_warp, colour_predictions, "Colour")

            # Display predictions
            cv2.imshow('Ensemble Predictions', ensemble_viz)
            cv2.imshow('Contour Predictions', contour_viz)
            cv2.imshow('Colour Predictions', colour_viz)
        else:
            cv2.imshow('Original', undistorted)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()