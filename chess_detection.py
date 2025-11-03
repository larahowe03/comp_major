import cv2
import pickle
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO
from collections import deque
import time
import copy
from ultralytics.engine.results import Boxes
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


class PredictionStabilizer:
    """
    Stabilize predictions using highest confidence over time window.
    """
    def __init__(self, window_seconds=1.0, max_distance=50):
        """
        Args:
            window_seconds: Time window for selecting best prediction (seconds)
            max_distance: Max distance (pixels) to consider same object
        """
        self.window_seconds = window_seconds
        self.max_distance = max_distance
        self.history = deque()  # [(timestamp, boxes, confs, cls)]
    
    def _clean_old_entries(self, current_time):
        """Remove entries older than the time window."""
        cutoff_time = current_time - self.window_seconds
        while self.history and self.history[0][0] < cutoff_time:
            self.history.popleft()
    
    def _calculate_distance(self, box1, box2):
        """Calculate center distance between two boxes."""
        center1 = [(box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2]
        center2 = [(box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2]
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def update(self, result):
        """
        Update with new predictions and return stabilized result.
        Uses highest confidence detection within the time window.
        
        Args:
            result: YOLO result object
            
        Returns:
            Stabilized YOLO result object with highest confidence predictions
        """
        current_time = time.time()
        
        # Clean old entries
        self._clean_old_entries(current_time)
        
        # Extract current predictions
        if len(result.boxes) == 0:
            return result
        
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        cls = result.boxes.cls.cpu().numpy()
        
        # Add to history
        self.history.append((current_time, boxes, confs, cls))
        
        # If not enough history, return original
        if len(self.history) < 2:
            return result
        
        # Find highest confidence predictions
        stabilized_boxes = []
        stabilized_confs = []
        stabilized_cls = []
        
        for i, (box, conf, c) in enumerate(zip(boxes, confs, cls)):
            # Find matching detections in history
            best_box = box
            best_conf = conf
            best_cls = c
            
            for hist_time, hist_boxes, hist_confs, hist_cls in list(self.history):
                # Find closest box of same class
                for j, (hist_box, hist_conf, hist_c) in enumerate(zip(hist_boxes, hist_confs, hist_cls)):
                    # Check same class
                    if hist_c != c:
                        continue
                    
                    # Check distance
                    dist = self._calculate_distance(box, hist_box)
                    if dist < self.max_distance:
                        # Update if this has higher confidence
                        if hist_conf > best_conf:
                            best_box = hist_box
                            best_conf = hist_conf
                            best_cls = hist_c
            
            stabilized_boxes.append(best_box)
            stabilized_confs.append(best_conf)
            stabilized_cls.append(best_cls)
        
        # Create new result with stabilized predictions
        if len(stabilized_boxes) == 0:
            return result
        
        stabilized_boxes_tensor = torch.tensor(np.array(stabilized_boxes), 
                                               device=result.boxes.xyxy.device, 
                                               dtype=torch.float32)
        stabilized_confs_tensor = torch.tensor(np.array(stabilized_confs), 
                                               device=result.boxes.conf.device, 
                                               dtype=torch.float32)
        stabilized_cls_tensor = torch.tensor(np.array(stabilized_cls), 
                                             device=result.boxes.cls.device, 
                                             dtype=torch.float32)
        
        # Stack all data
        boxes_data = torch.cat([
            stabilized_boxes_tensor,
            stabilized_confs_tensor.unsqueeze(1),
            stabilized_cls_tensor.unsqueeze(1)
        ], dim=1)
        
        # Create new Boxes object
        from ultralytics.engine.results import Boxes
        new_boxes = Boxes(boxes_data, result.orig_shape)
        
        # Clone result and replace boxes
        stabilized_result = copy.deepcopy(result)
        stabilized_result.boxes = new_boxes
        
        return stabilized_result


def contour_model_prediction(img):
    result = contour_model.predict(
        source=img,
        conf=0.1,
        imgsz=448,
        device=device,
        verbose=False
    )
    return result[0]


def colour_model_prediction(img):
    result = colour_model.predict(
        source=img,
        conf=0.1,
        imgsz=448,
        device=device,
        verbose=False
    )
    return result[0]


def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two boxes."""
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
    
    return inter_area / union_area


def get_piece_type(class_name):
    """Extract piece type from class name."""
    if isinstance(class_name, str):
        if '_' in class_name:
            return class_name.split('_')[1]
        return class_name
    return class_name

def debug_ensemble_analysis(contour_result, colour_result, iou_threshold=0.5):
    """
    Inspect how contour and colour detections overlap before ensembling.
    Prints IoU values, class mismatches, and potential merge candidates.
    """
    contour_boxes = contour_result.boxes
    colour_boxes = colour_result.boxes

    if len(contour_boxes) == 0 or len(colour_boxes) == 0:
        print("[⚠️] One of the models returned no detections!")
        return

    contour_xyxy = contour_boxes.xyxy.cpu().numpy()
    contour_conf = contour_boxes.conf.cpu().numpy()
    contour_cls = contour_boxes.cls.cpu().numpy()
    contour_names = contour_result.names

    colour_xyxy = colour_boxes.xyxy.cpu().numpy()
    colour_conf = colour_boxes.conf.cpu().numpy()
    colour_cls = colour_boxes.cls.cpu().numpy()
    colour_names = colour_result.names

    print("\n[DEBUG] Analysing overlap between models...")
    print(f"  Contour boxes: {len(contour_boxes)} | Colour boxes: {len(colour_boxes)}")

    for i, cbox in enumerate(colour_xyxy):
        cname = colour_names[int(colour_cls[i])]
        cconf = colour_conf[i]
        print(f"\n🟦 Colour {i}: {cname} (conf={cconf:.2f})")
        print("   Matches:")

        for j, kbox in enumerate(contour_xyxy):
            kname = contour_names[int(contour_cls[j])]
            kconf = contour_conf[j]

            iou = calculate_iou(cbox, kbox)
            same_class = (cname == kname)

            # Generate result message
            if same_class and iou > iou_threshold:
                msg = f"      ✅ Contour {j}: {kname} (conf={kconf:.2f}) — IoU={iou:.2f} [MERGE]"
            elif same_class and iou <= iou_threshold:
                msg = f"      ❌ Contour {j}: {kname} (conf={kconf:.2f}) — IoU={iou:.2f} [low overlap]"
            elif not same_class and iou > iou_threshold:
                msg = f"      ⚠️ Contour {j}: {kname} (conf={kconf:.2f}) — IoU={iou:.2f} [class mismatch!]"
            else:
                msg = f"      · Contour {j}: {kname} (conf={kconf:.2f}) — IoU={iou:.2f}"

            print(msg)

    print("\n[END DEBUG] Ensemble analysis complete.\n")

def ensemble_predictions(contour_result, colour_result, iou_threshold=0.5, conf_weight_contour=0.5):
    """Ensemble predictions from contour and color models."""
    contour_boxes = contour_result.boxes
    colour_boxes = colour_result.boxes
    
    if len(contour_boxes) == 0:
        return colour_result
    if len(colour_boxes) == 0:
        return contour_result
    
    contour_names = contour_result.names
    colour_names = colour_result.names
    
    contour_xyxy = contour_boxes.xyxy.cpu().numpy()
    contour_conf = contour_boxes.conf.cpu().numpy()
    contour_cls = contour_boxes.cls.cpu().numpy()
    
    colour_xyxy = colour_boxes.xyxy.cpu().numpy()
    colour_conf = colour_boxes.conf.cpu().numpy()
    colour_cls = colour_boxes.cls.cpu().numpy()
    
    all_detections = []
    
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
    
    final_boxes = []
    final_confs = []
    final_cls = []
    used_indices = set()
    
    for i in range(len(all_detections)):
        if i in used_indices:
            continue
        
        current = all_detections[i]
        overlapping = [current]
        overlapping_indices = [i]
        
        for j in range(i + 1, len(all_detections)):
            if j in used_indices:
                continue
            
            other = all_detections[j]
            
            if current['piece_type'] != other['piece_type']:
                continue
            
            iou = calculate_iou(current['box'], other['box'])
            
            if iou > iou_threshold:
                overlapping.append(other)
                overlapping_indices.append(j)
                used_indices.add(j)
        
        used_indices.add(i)
        
        if len(overlapping) > 1:
            boxes = np.array([d['box'] for d in overlapping])
            weights = np.array([d['weight'] for d in overlapping])
            weights = weights / weights.sum()
            avg_box = np.average(boxes, axis=0, weights=weights)
            
            confs = np.array([d['conf'] for d in overlapping])
            avg_conf = np.average(confs, weights=weights)
            avg_conf = min(1.0, avg_conf * 1.2)
            
            color_detections = [d for d in overlapping if d['source'] == 'colour']
            if color_detections:
                best_color = max(color_detections, key=lambda x: x['conf'])
                final_cls_id = best_color['cls_id']
            else:
                final_cls_id = overlapping[0]['cls_id']
            
            final_boxes.append(avg_box)
            final_confs.append(avg_conf)
            final_cls.append(final_cls_id)
        else:
            final_boxes.append(current['box'])
            final_confs.append(current['conf'])
            final_cls.append(current['cls_id'])
    
    if len(final_boxes) == 0:
        return colour_result
    
    final_boxes_tensor = torch.tensor(np.array(final_boxes), device=colour_boxes.xyxy.device, dtype=torch.float32)
    final_confs_tensor = torch.tensor(np.array(final_confs), device=colour_boxes.conf.device, dtype=torch.float32)
    final_cls_tensor = torch.tensor(np.array(final_cls), device=colour_boxes.cls.device, dtype=torch.float32)
    
    boxes_data = torch.cat([
        final_boxes_tensor,
        final_confs_tensor.unsqueeze(1),
        final_cls_tensor.unsqueeze(1)
    ], dim=1)
    
    new_boxes = Boxes(boxes_data, colour_result.orig_shape)
    
    ensemble_result = copy.deepcopy(colour_result)
    ensemble_result.boxes = new_boxes
    
    return ensemble_result


def ensemble_model(contoured_warp, coloured_warp):
    """Run both models and ensemble their predictions."""
    contour_predictions = contour_model_prediction(contoured_warp)
    colour_predictions = colour_model_prediction(coloured_warp)
    
    debug_ensemble_analysis(contour_predictions, colour_predictions, iou_threshold=0.5)
    
    ensemble_result = ensemble_predictions(
        contour_predictions, 
        colour_predictions,
        iou_threshold=0.5,
        conf_weight_contour=0.5
    )
    
    return ensemble_result


def undistort(img, K, d):
    return cv2.undistort(img, K, d, None, K)


# Initialize camera and calibration
def initialize_camera(phone_ip="192.168.0.105", port="4747"):
    """Initialize camera connection."""
    urls = [
        f"http://{phone_ip}:{port}/video",
        f"http://{phone_ip}:{port}/mjpegfeed",
    ]
    
    for url in urls:
        print(f"Trying: {url}")
        cap = cv2.VideoCapture(url)
        if cap.isOpened():
            print(f"Connected successfully to {url}")
            return cap
        cap.release()
    
    print("Could not connect to camera")
    return None


def load_calibration(filepath="calibration_coefficients.pkl"):
    """Load camera calibration data."""
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data['camera_matrix'], data['distortion_coeffs']


# Global variables for camera and calibration
camera = None
K_matrix = None
dist_coeffs = None
stabilizer = None


def init_detection_system():
    """Initialize the detection system (call once at startup)."""
    global camera, K_matrix, dist_coeffs, stabilizer
    
    camera = initialize_camera()
    if camera is None:
        return False
    
    K_matrix, dist_coeffs = load_calibration()
    stabilizer = PredictionStabilizer(window_seconds=1.0, max_distance=50)
    
    return True


def get_current_frame():
    """Get and process current frame from camera."""
    global camera, K_matrix, dist_coeffs, stabilizer
    
    if camera is None:
        return None, None, None
    
    ret, frame = camera.read()
    if not ret:
        return None, None, None
    
    # Undistort
    undistorted = undistort(frame, K_matrix, dist_coeffs)
    
    # Warp board
    warped, contoured_img, pts_src = process_chess_image(undistorted)
    
    if warped is None:
        return undistorted, None, None, None
    
    return undistorted, warped, pts_src


def detect_pieces(warped):
    """Detect chess pieces on warped board."""
    global stabilizer
    
    if warped is None:
        return None
    
    # Prepare images
    contoured_warp = preprocess_image(warped)
    coloured_warp = warped
        
    # Get ensemble predictions
    ensemble_result= ensemble_model(contoured_warp, coloured_warp)
    
    # Apply stabilization
    stabilized_result = stabilizer.update(ensemble_result)
    
    # Extract boxes from stabilized result
    stabilized_boxes = []
    for box in stabilized_result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        confidence = box.conf[0].cpu().numpy()
        class_id = int(box.cls[0].cpu().numpy())
        
        stabilized_boxes.append({
            'box': [float(x1), float(y1), float(x2), float(y2)],
            'bottom': float(y2),
            'confidence': float(confidence),
            'class_id': class_id,
            'class_name': stabilized_result.names[class_id]
        })
    
    return stabilized_result


def cleanup_camera():
    """Clean up camera resources."""
    global camera
    if camera is not None:
        camera.release()
        cv2.destroyAllWindows()