from allowable_moves import Piece
import gui
from chess_detection import get_current_frame, detect_pieces, cleanup_camera, init_detection_system
import cv2
from corner_variation import calculate_corner_variation

# State definitions
IDLE = 1
STATIC = 2
MOVING = 3
PREDICT = 4
CHANGED = 5
UPDATE_BOARD = 6

current_state = IDLE
prev_state = IDLE
def get_cell_location(unmargined_img, final_preds, bottom_loc):
    # Get image dimensions
    h, w = unmargined_img.shape[:2]
    
    # Calculate cell size (assuming square board)
    cell_width = w / 8
    cell_height = h / 8
    
    # List to store pieces with their positions
    pieces_with_positions = []
    
    # Process each detected piece
    for pred, loc in zip(final_preds, bottom_loc):
        if loc is None or len(loc) != 2:
            continue
        
        x, y = loc
        
        # Check if location is within valid board area (with some tolerance)
        if x < -cell_width or x >= w + cell_width or y < -cell_height or y >= h + cell_height:
            # print(f"Warning: Piece at ({x}, {y}) is too far outside board bounds")
            continue
        
        # Determine which column this piece belongs to
        col = int(x / cell_width)
        col = max(0, min(7, col))
        
        # Find the closest row by finding which row boundary (bottom edge) is closest
        min_gap = float('inf')
        row = 0
        
        for i in range(8):
            # Calculate the bottom y-coordinate of row i (the gap/boundary)
            row_bottom_y = (i + 1) * cell_height
            
            # Calculate absolute distance from piece to this row's bottom boundary
            gap = abs(row_bottom_y - y)
            
            if gap < min_gap:
                min_gap = gap
                row = i
        
        # Get chess notation for logging
        notation = get_chess_notation(row, col)
        
        # Add to pieces list
        pieces_with_positions.append({
            'row': row,
            'col': col,
            'class_name': pred
        })
        
        # print(f"Detected {pred} at {notation} (row={row}, col={col}, pixel=({x:.1f}, {y:.1f}), min_gap={min_gap:.1f})")
    
    # Convert to board state using existing function
    board_state = detections_to_board(pieces_with_positions)
    
    return board_state
    


def detections_to_board(pieces_with_positions):
    board = [[None for _ in range(8)] for _ in range(8)]

    for piece in pieces_with_positions:
        row, col = piece['row'], piece['col']
        cls_name = piece['class_name'].lower().strip()  # ensure consistent formatting

        # Split into type and color
        if "white" in cls_name:
            color = "white"
            piece_type = cls_name.replace("white_", "")
        elif "black" in cls_name:
            color = "black"
            piece_type = cls_name.replace("black_", "")
        else:
            continue

        # 🔧 Normalize naming to match allowable_moves.Piece conventions
        if piece_type == "rook":
            piece_type = "castle"
        elif piece_type == "queen":
            piece_type = "queen"  # keep as is
        elif piece_type == "king":
            piece_type = "king"
        elif piece_type == "bishop":
            piece_type = "bishop"
        elif piece_type == "knight":
            piece_type = "knight"
        elif piece_type == "pawn":
            piece_type = "pawn"

        # Create Piece instance and store
        board[row][col] = Piece(piece_type, color)

    return board

def get_chess_notation(row, col, flip_board=False):
    """Convert row/col to chess notation (e.g., 'e4')."""
    if flip_board:
        row = 7 - row
        col = 7 - col
    
    files = 'abcdefgh'
    ranks = '87654321'  # Assuming row 0 is rank 8
    
    return files[col] + ranks[row]

def calculate_overlap(box1, box2):
    """Calculate the intersection area between two boxes."""
    # box format: {"x1": ..., "y1": ..., "x2": ..., "y2": ...}
    x1_inter = max(box1["x1"], box2["x1"])
    y1_inter = max(box1["y1"], box2["y1"])
    x2_inter = min(box1["x2"], box2["x2"])
    y2_inter = min(box1["y2"], box2["y2"])
    
    # If no intersection, return 0
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0
    
    intersection_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    return intersection_area

def stabilise_piece_prediction(contour_boxes, colour_boxes, contour_classes, colour_classes, piece_colours, previous_preds=None, previous_sources=None, previous_boxes=None):
    """
    Match contour boxes with colour boxes based on overlap and validate colour predictions.
    PRIORITY: Use colour model predictions when they match HSV-detected piece colour.
    Hold colour predictions across frames to reduce flutter.
    """
    if previous_preds is None:
        previous_preds = []
    if previous_sources is None:
        previous_sources = []
    if previous_boxes is None:
        previous_boxes = []
    
    # print(f"\n=== STABILISE_PIECE_PREDICTION DEBUG ===")
    # print(f"Contour boxes: {len(contour_boxes)}")
    # print(f"Colour boxes: {len(colour_boxes)}")
    # print(f"Piece colours (HSV-detected): {piece_colours}")
    # print(f"Contour classes: {contour_classes}")
    # print(f"Colour classes: {colour_classes}")
    
    final_boxes = []
    final_preds = []
    final_sources = []
    
    # Build a lookup for previous predictions by approximate position
    previous_lookup = {}
    for idx, prev_box in enumerate(previous_boxes):
        if idx < len(previous_preds):
            # Use center of box as key (rounded to nearest 30 pixels for matching)
            center_x = ((prev_box["x1"] + prev_box["x2"]) // 2) // 30 * 30
            center_y = ((prev_box["y1"] + prev_box["y2"]) // 2) // 30 * 30
            previous_lookup[(center_x, center_y)] = {
                'pred': previous_preds[idx],
                'source': previous_sources[idx]
            }
    
    for i, (contour_box, contour_pred) in enumerate(zip(contour_boxes, contour_classes)):
        # print(f"\n--- Processing contour box {i} ---")
        # print(f"  Contour pred: {contour_pred}")
        
        # Ensure box is a dict
        if not isinstance(contour_box, dict):
            # print(f"  ERROR: contour_box is not a dict! Type: {type(contour_box)}")
            continue
        
        hsv_piece_colour = piece_colours[i] if i < len(piece_colours) else None
        # print(f"  HSV detected piece colour: {hsv_piece_colour}")
        
        # Find matching colour box with maximum overlap
        best_overlap = 0
        best_colour_pred = None
        
        for j, (colour_box, colour_pred) in enumerate(zip(colour_boxes, colour_classes)):
            overlap = calculate_overlap(contour_box, colour_box)
            
            if overlap > best_overlap:
                best_overlap = overlap
                best_colour_pred = colour_pred
                # print(f"  Found colour match with box {j}: overlap={overlap:.2f}, pred={colour_pred}")
        
        # print(f"  Best overlap: {best_overlap:.2f}, Best colour pred: {best_colour_pred}")
        
        # Get previous prediction for this position
        center_x = ((contour_box["x1"] + contour_box["x2"]) // 2) // 30 * 30
        center_y = ((contour_box["y1"] + contour_box["y2"]) // 2) // 30 * 30
        prev_data = previous_lookup.get((center_x, center_y), None)
        
        # DECISION LOGIC: Prioritize colour predictions when valid
        
        # Option 1: Current frame has valid colour prediction that matches HSV
        if best_overlap > 0 and best_colour_pred is not None and hsv_piece_colour is not None:
            # Extract piece colour from prediction (e.g., "white_pawn" -> "white")
            pred_piece_colour = best_colour_pred.split('_')[0].lower()
            # print(f"  Colour model says: {pred_piece_colour}, HSV says: {hsv_piece_colour}")
            
            # Validate: colour prediction should match HSV detection
            colour_match = (pred_piece_colour == hsv_piece_colour)
            
            if colour_match:
                final_boxes.append(contour_box)
                final_preds.append(best_colour_pred)
                final_sources.append('colour')
                # print(f"  ✓ USING CURRENT COLOUR PREDICTION (MATCHES HSV): {best_colour_pred}")
                continue
            # else:
                # print(f"  ✗ COLOUR MISMATCH: model={pred_piece_colour}, HSV={hsv_piece_colour}")
        
        # Option 2: Hold previous colour prediction (REDUCE FLUTTER)
        if prev_data is not None and prev_data['source'] == 'colour':
            # Optionally validate that previous prediction still makes sense
            # For now, trust it to reduce flutter
            final_boxes.append(contour_box)
            final_preds.append(prev_data['pred'])
            final_sources.append('colour')
            # print(f"  ✓ HOLDING PREVIOUS COLOUR PREDICTION: {prev_data['pred']}")
            continue
        
        # Option 3: Fall back to contour prediction with HSV colour
        final_pred = f"{hsv_piece_colour}_{contour_pred}" if hsv_piece_colour else contour_pred
        final_boxes.append(contour_box)
        final_preds.append(final_pred)
        final_sources.append('contour')
    #     print(f"  ⚠ USING CONTOUR PREDICTION: {final_pred}")
    
    # print(f"\n=== FINAL RESULTS ===")
    # print(f"Final preds: {final_preds}")
    # print(f"Final sources: {final_sources}")
    
    return final_boxes, final_preds, final_sources
def visualise_detections(img, boxes, predictions, bottom_loc, class_names=None):
    annotated = img.copy()
    
    colors = [
        (0, 255, 0),      # Green
        (0, 0, 255),      # Red
        (255, 0, 0),      # Blue
        (255, 255, 0),    # Cyan
        (255, 0, 255),    # Magenta
        (0, 255, 255),    # Yellow
    ]
    
    for i, (box, pred) in enumerate(zip(boxes, predictions)):
        x1 = box["x1"]
        y1 = box["y1"]
        x2 = box["x2"]
        y2 = box["y2"]

        # Pick color based on index
        color = colors[i % len(colors)]
        
        # Draw rectangle
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{pred}" if pred is not None else "Unknown"
        cv2.putText(annotated, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Draw bottom locations as dots
    if bottom_loc is not None:
        for i, loc in enumerate(bottom_loc):
            if loc is not None and len(loc) == 2:
                x, y = loc
                # Draw a filled circle at the bottom location
                cv2.circle(annotated, (int(x), int(y)), radius=5, color=(255, 0, 255), thickness=-1)
                # Draw a small circle outline for better visibility
                cv2.circle(annotated, (int(x), int(y)), radius=6, color=(255, 255, 255), thickness=1)
    
    return annotated

board_state = [
    [Piece("castle", "white"), Piece("knight", "white"), Piece("bishop", "white"), Piece("queen", "white"), Piece("king", "white"), Piece("bishop", "white"), Piece("knight", "white"), Piece("castle", "white")],
    [Piece("pawn", "white"), Piece("pawn", "white"), Piece("pawn", "white"), Piece("pawn", "white"), Piece("pawn", "white"), Piece("pawn", "white"), Piece("pawn", "white"), Piece("pawn", "white")],
    [None, None, None, None, None, None, None, None],
    [None, None, None, None, None, None, None, None],
    [None, None, None, None, None, None, None, None],
    [None, None, None, None, None, None, None, None],
    [Piece("pawn", "black"), Piece("pawn", "black"), Piece("pawn", "black"), Piece("pawn", "black"), Piece("pawn", "black"), Piece("pawn", "black"), Piece("pawn", "black"), Piece("pawn", "black")],
    [Piece("castle", "black"), Piece("knight", "black"), Piece("bishop", "black"), Piece("queen", "black"), Piece("king", "black"), Piece("bishop", "black"), Piece("knight", "black"), Piece("castle", "black")]
]

def transform_boxes_remove_margin(boxes, margin=80):
    """Transform box coordinates from margined image to unmargined image."""
    transformed_boxes = []
    bottom_loc = []
    for box in boxes:
        x1 = box["x1"] - 63
        y1 = box["y1"] - 70
        x2 = box["x2"] - 63
        y2 = box["y2"] - 70
        transformed_box = {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2
        }
        x_coord = (x1 + (x2-x1)//2)
        y_coord = y2
        bottom_loc.append([x_coord, y_coord])
        transformed_boxes.append(transformed_box)
    return transformed_boxes, bottom_loc

# Initialize detection system
# print("Initializing chess detection system...")
if not init_detection_system():
    # print("Failed to initialize detection system")
    exit()

# print("System initialized. Starting main loop...")

running = True
detection_result = None
warped_board = None

pts_src_buffer = []

MARGIN = 80

final_preds = None
final_sources = None

prev_board_state = board_state

while running:    
    # Get current frame from camera
    warp_margined, warp_unmargined, contoured_img, pts_src = get_current_frame()

    pts_src_buffer.append(pts_src)

    if len(pts_src_buffer) > 20:
        pts_src_buffer.pop(0)

    is_stable, max_std, details = calculate_corner_variation(pts_src_buffer, threshold=10.0)

    # print("is_stable", is_stable)
    print("current_state", current_state)
        
    if warp_margined is not None:
        cv2.imshow('warp_margined', warp_margined)
        cv2.imshow('warp_unmargined', warp_unmargined)
                
        # Detect pieces
        contour_boxes, colour_boxes, contour_classes, colour_classes, contour_annotated, colour_annotated, piece_colours = detect_pieces(warp_margined)
                
        cv2.imshow('contour_annotated', contour_annotated)
        cv2.imshow('colour_annotated', colour_annotated)

        h, w = colour_annotated.shape[:2]

        contour_cropped = contour_annotated[MARGIN:h-MARGIN, MARGIN:w-MARGIN]
        colour_cropped = colour_annotated[MARGIN:h-MARGIN, MARGIN:w-MARGIN]

        cv2.imshow('contour_cropped', contour_cropped)
        cv2.imshow('colour_cropped', colour_cropped)

        final_boxes, final_preds, final_sources = stabilise_piece_prediction(contour_boxes, colour_boxes, contour_classes, colour_classes, piece_colours, final_preds, final_sources)

        # Transform boxes to unmargined coordinates
        final_boxes_unmargined, bottom_loc = transform_boxes_remove_margin(final_boxes, MARGIN)

        # Visualize on unmargined image
        annotated_img_unmargined = visualise_detections(warp_unmargined, final_boxes_unmargined, final_preds, bottom_loc)
        cv2.imshow('annotated_img_unmargined', cv2.cvtColor(annotated_img_unmargined, cv2.COLOR_BGR2RGB))
        
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False
    
    # Always update GUI
    running = gui.do_gui(board_state, prev_board_state, current_state)

    # State machine logic
    if current_state == IDLE:
        # Wait for board setup button
        if gui.setup_mode:
            prev_state = current_state
            current_state = STATIC
            # print("Game beginning - transitioning to STATIC state")
    
    elif current_state == STATIC:
        # Monitor board for changes
        board_state = get_cell_location(warp_unmargined, final_preds, bottom_loc)
        prev_board_state = board_state

        if is_stable == False:
            current_state = MOVING
    
    elif current_state == PREDICT:
        # Use detection results to predict move
        if detection_result is not None:
            # TODO: Convert detection_result to board state
            pass
        
        # Transition to next state
        # current_state = UPDATE_BOARD
    
    elif current_state == MOVING:
        # Detect piece movement
        if is_stable == True:
            current_state = PREDICT
    
    elif current_state == CHANGED:
        # Board state has changed
        # TODO: Validate the change
        # current_state = UPDATE_BOARD
        pass
    
    elif current_state == UPDATE_BOARD:
        # Update the board_state array
        if detection_result is not None:
            # TODO: Update board_state based on detection_result
            # Example:
            # board_state = convert_detections_to_board(detection_result)
            pass
        
        # Transition back to monitoring
        # current_state = STATIC

# Cleanup
# print("Cleaning up...")
cleanup_camera()
gui.cleanup()