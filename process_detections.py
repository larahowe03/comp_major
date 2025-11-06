import cv2

# ------------------------------------------------------------------------
# PROCESS DETECTIONS FUNCTIONS
# Contains functions related to model predictions and bounding boxes
# ------------------------------------------------------------------------

def calculate_overlap(box1, box2):
    """
    Calculate the area of overlap in contour and colour bounding boxes
    Return overlapping area if there is overlap.
    """
    x1_inter = max(box1["x1"], box2["x1"])
    y1_inter = max(box1["y1"], box2["y1"])
    x2_inter = min(box1["x2"], box2["x2"])
    y2_inter = min(box1["y2"], box2["y2"])
    
    # exit if there is no overlap
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0
    
    # else return the overlap area
    return (x2_inter - x1_inter) * (y2_inter - y1_inter)


def stabilise_piece_prediction(contour_boxes, colour_boxes, contour_classes, colour_classes, 
                               piece_colours, previous_preds=None, previous_sources=None, 
                               previous_boxes=None):
    """
    Stabalised predictions to make sure the label or position does not fluctuate
    """
    
    if previous_preds is None:
        previous_preds = []
    if previous_sources is None:
        previous_sources = []
    if previous_boxes is None:
        previous_boxes = []
    
    final_boxes = []
    final_preds = []
    final_sources = []
    
    # Build lookup dictionary once
    previous_lookup = {}
    for idx, prev_box in enumerate(previous_boxes):
        if idx < len(previous_preds):
            center_x = ((prev_box["x1"] + prev_box["x2"]) // 2) // 30 * 30
            center_y = ((prev_box["y1"] + prev_box["y2"]) // 2) // 30 * 30
            previous_lookup[(center_x, center_y)] = {
                'pred': previous_preds[idx],
                'source': previous_sources[idx]
            }
    
    # Process each contour box
    for i, (contour_box, contour_pred) in enumerate(zip(contour_boxes, contour_classes)):

        if not isinstance(contour_box, dict):
            continue
        
        hsv_piece_colour = piece_colours[i] if i < len(piece_colours) else None
        
        # Find best colour match
        best_overlap = 0
        best_colour_pred = None
        
        for colour_box, colour_pred in zip(colour_boxes, colour_classes):
            overlap = calculate_overlap(contour_box, colour_box)
            if overlap > best_overlap:
                best_overlap = overlap
                best_colour_pred = colour_pred
        
        # Get previous prediction
        center_x = ((contour_box["x1"] + contour_box["x2"]) // 2) // 30 * 30
        center_y = ((contour_box["y1"] + contour_box["y2"]) // 2) // 30 * 30
        prev_data = previous_lookup.get((center_x, center_y))
        
        # valid current colour prediction
        if best_overlap > 0 and best_colour_pred and hsv_piece_colour:
            pred_piece_colour = best_colour_pred.split('_')[0].lower()
            
            if pred_piece_colour == hsv_piece_colour:
                final_boxes.append(contour_box)
                final_preds.append(best_colour_pred)
                final_sources.append('colour')
                continue
        
        # hold previous colour prediction
        if prev_data and prev_data['source'] == 'colour':
            final_boxes.append(contour_box)
            final_preds.append(prev_data['pred'])
            final_sources.append('colour')
            continue
        
        # use contour prediction
        final_pred = f"{hsv_piece_colour}_{contour_pred}" if hsv_piece_colour else contour_pred
        final_boxes.append(contour_box)
        final_preds.append(final_pred)
        final_sources.append('contour')
    
    return final_boxes, final_preds, final_sources


def transform_boxes_remove_margin(boxes, margin_x=63, margin_y=70):
    """
    Transfrom the boxes and to get bottom locations in the unmargined image
    """
    
    transformed_boxes = []
    bottom_loc = []
    
    for box in boxes:
        x1 = box["x1"] - margin_x
        y1 = box["y1"] - margin_y
        x2 = box["x2"] - margin_x
        y2 = box["y2"] - margin_y
        
        transformed_boxes.append({
            "x1": x1, "y1": y1,
            "x2": x2, "y2": y2
        })
        
        # Calculate bottom center
        bottom_loc.append([x1 + (x2 - x1) // 2, y2])
    
    return transformed_boxes, bottom_loc


def visualise_detections(img, boxes, predictions, bottom_loc):
    """
    Visualisation function 
    """
    
    annotated = img.copy()
    
    colors = [
        (0, 255, 0), (0, 0, 255), (255, 0, 0),
        (255, 255, 0), (255, 0, 255), (0, 255, 255)
    ]
    
    # Draw boxes and labels
    for i, (box, pred) in enumerate(zip(boxes, predictions)):
        color = colors[i % 6]
        
        cv2.rectangle(annotated, (box["x1"], box["y1"]), (box["x2"], box["y2"]), color, 2)
        
        label = pred if pred else "Unknown"
        cv2.putText(annotated, label, (box["x1"], box["y1"] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Draw bottom locations
    if bottom_loc:
        for loc in bottom_loc:
            if loc and len(loc) == 2:
                x, y = int(loc[0]), int(loc[1])
                cv2.circle(annotated, (x, y), 5, (255, 0, 255), -1)
                cv2.circle(annotated, (x, y), 6, (255, 255, 255), 1)
    
    return annotated