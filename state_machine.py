from allowable_moves import Piece, detect_move
import gui
from chess_detection import get_current_frame, detect_pieces, cleanup_camera, init_detection_system, make_clahe
import cv2
from corner_variation import calculate_corner_variation
import numpy as np
from collections import Counter, deque
import time

# State definitions
IDLE = 1
STATIC = 2
MOVING = 3
PREDICT = 4
CHANGED = 5
UPDATE_BOARD = 6
INVALID = 7  # New state for invalid moves

current_state = IDLE
prev_state = IDLE

# Prediction voting system
VOTING_PERIOD = 1.0  # seconds
STABILITY_DELAY = 2.0  # seconds - delay before transitioning from MOVING to PREDICT
prediction_history = deque(maxlen=150)  # ~5 seconds at 30fps
voting_start_time = None
stability_start_time = None

# Invalid move tracking
invalid_move_info = None  # Stores the invalid move details

def get_most_common_board_state(prediction_history):
    """
    Extract the most common board state from prediction history.
    Returns None if no valid predictions exist.
    """
    if not prediction_history:
        return None
    
    # Count occurrences of each board state (convert to hashable format)
    board_state_counts = Counter()
    
    for board_state in prediction_history:
        # Convert board state to hashable tuple representation
        board_tuple = tuple(
            tuple(
                (piece.type, piece.colour) if piece is not None else None
                for piece in row
            )
            for row in board_state
        )
        board_state_counts[board_tuple] += 1
    
    # Get most common board state
    if not board_state_counts:
        return None
    
    most_common_tuple, count = board_state_counts.most_common(1)[0]
    
    # Convert back to board state format
    board_state = []
    for row_tuple in most_common_tuple:
        row = []
        for cell in row_tuple:
            if cell is None:
                row.append(None)
            else:
                piece_type, piece_colour = cell
                row.append(Piece(piece_type, piece_colour))
        board_state.append(row)
    
    return board_state

def get_cell_location(unmargined_img, final_preds, bottom_loc):
    """Optimized cell location detection with cached calculations"""
    if unmargined_img is None or final_preds is None or bottom_loc is None:
        return None
    
    h, w = unmargined_img.shape[:2]
    
    # Calculate cell size once
    cell_width = w / 8
    cell_height = h / 8
    
    # Pre-calculate row boundaries for faster lookup
    row_boundaries = np.array([(i + 1) * cell_height for i in range(8)])
    
    pieces_with_positions = []
    
    for pred, loc in zip(final_preds, bottom_loc):
        if loc is None or len(loc) != 2:
            continue
        
        x, y = loc
        
        # Quick bounds check
        if not (-cell_width <= x < w + cell_width and -cell_height <= y < h + cell_height):
            continue
        
        # Determine column (fast integer division)
        col = max(0, min(7, int(x / cell_width)))
        
        # Find closest row using vectorized numpy operation
        distances = np.abs(row_boundaries - y)
        row = int(np.argmin(distances))
        
        pieces_with_positions.append({
            'row': row,
            'col': col,
            'class_name': pred
        })
    
    # Convert to board state
    board_state = detections_to_board(pieces_with_positions)
    
    return board_state

def detections_to_board(pieces_with_positions):
    """Optimized board creation with early validation"""
    board = [[None for _ in range(8)] for _ in range(8)]
    
    # Piece type mapping (avoid repeated string operations)
    type_mapping = {
        "rook": "castle",
        "queen": "queen",
        "king": "king",
        "bishop": "bishop",
        "knight": "knight",
        "pawn": "pawn"
    }

    for piece in pieces_with_positions:
        row, col = piece['row'], piece['col']
        cls_name = piece['class_name'].lower().strip()

        # Determine color and piece type
        if "white" in cls_name:
            color = "white"
            piece_type = cls_name.replace("white_", "")
        elif "black" in cls_name:
            color = "black"
            piece_type = cls_name.replace("black_", "")
        else:
            continue

        # Normalize naming using mapping
        piece_type = type_mapping.get(piece_type, piece_type)
        
        # Create and store piece
        board[row][col] = Piece(piece_type, color)

    return board

def get_chess_notation(row, col, flip_board=False):
    """Convert row/col to chess notation (e.g., 'e4')."""
    if flip_board:
        row = 7 - row
        col = 7 - col
    
    files = 'abcdefgh'
    ranks = '87654321'
    
    return files[col] + ranks[row]

def calculate_overlap(box1, box2):
    """Optimized overlap calculation"""
    x1_inter = max(box1["x1"], box2["x1"])
    y1_inter = max(box1["y1"], box2["y1"])
    x2_inter = min(box1["x2"], box2["x2"])
    y2_inter = min(box1["y2"], box2["y2"])
    
    # Early exit if no overlap
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0
    
    return (x2_inter - x1_inter) * (y2_inter - y1_inter)

def has_board_changed(prev_board, current_board):
    if prev_board is None or current_board is None:
        return False
    
    for row in range(8):
        for col in range(8):
            prev_piece = prev_board[row][col]
            curr_piece = current_board[row][col]
            
            # Check if occupancy changed
            prev_occupied = prev_piece is not None
            curr_occupied = curr_piece is not None
            
            # If one square is empty and the other isn't, board changed
            if prev_occupied != curr_occupied:
                return True
    
    return False

def check_piece_moved_from_invalid_square(prev_board, current_board, invalid_square):
    """Check if the piece on the invalid square has been moved"""
    if prev_board is None or current_board is None or invalid_square is None:
        return False
    
    row, col = invalid_square
    prev_piece = prev_board[row][col]
    curr_piece = current_board[row][col]
    
    # If there was a piece and now there isn't, it moved
    if prev_piece is not None and curr_piece is None:
        return True
    
    return False

def find_king(board, colour):
    """Find the position of a king on the board"""
    if board is None:
        return None
    
    for row in range(8):
        for col in range(8):
            piece = board[row][col]
            if piece is not None and piece.type == "king" and piece.colour == colour:
                return (row, col)
    
    return None

def get_squares_attacked_by_opponent(board, colour):
    """Get all squares attacked by opponent"""
    from allowable_moves import get_allowable_move, in_bounds
    
    not_allowable_moves = []

    for r in range(len(board)):
        for c in range(len(board[r])):
            if not in_bounds(r, c):
                continue
            target = board[r][c]
            if target is None or target.type == "king":
                continue
            if colour != target.colour:
                not_allowable_moves.extend(get_allowable_move(board, r, c))
    
    return not_allowable_moves

def check_if_checkmate(board, colour):
    """
    Naive checkmate detection: if the king is in check and has no legal moves.
    Returns (bool, list_of_attackers)
    """
    from allowable_moves import get_allowable_move, in_bounds
    king_pos = find_king(board, colour)
    if king_pos is None:
        return False, []

    attacked_squares = get_squares_attacked_by_opponent(board, colour)
    attackers = []

    # Find which pieces are attacking the king
    for r in range(8):
        for c in range(8):
            piece = board[r][c]
            if piece is not None and piece.colour != colour:
                moves = get_allowable_move(board, r, c)
                if list(king_pos) in moves:
                    attackers.append((r, c))

    # Basic checkmate test (no king escape)
    if list(king_pos) in attacked_squares:
        # Check if king has any valid escape squares
        from allowable_moves import get_allowable_move
        king_moves = get_allowable_move(board, king_pos[0], king_pos[1])
        safe_moves = [m for m in king_moves if m not in attacked_squares]
        if not safe_moves:
            return True, attackers

    return False, attackers


def check_if_in_check(board):
    """
    Check if either king is in check after a move.
    
    Returns:
        dict: {
            'white_in_check': bool,
            'black_in_check': bool,
            'message': str or None
        }
    """
    if board is None:
        return {
            'white_in_check': False,
            'black_in_check': False,
            'message': None
        }
    
    white_in_check = False
    black_in_check = False
    message = None
    
    # Find white king
    white_king_pos = find_king(board, "white")
    if white_king_pos is not None:
        white_attacked_squares = get_squares_attacked_by_opponent(board, "white")
        if list(white_king_pos) in white_attacked_squares:
            white_in_check = True
            print("⚠️  WHITE KING IS IN CHECK!")
            message = "White King is in check!"
    
    # Find black king
    black_king_pos = find_king(board, "black")
    if black_king_pos is not None:
        black_attacked_squares = get_squares_attacked_by_opponent(board, "black")
        if list(black_king_pos) in black_attacked_squares:
            black_in_check = True
            print("⚠️  BLACK KING IS IN CHECK!")
            message = "Black King is in check!"
    
    return {
        'white_in_check': white_in_check,
        'black_in_check': black_in_check,
        'message': message
    }

def stabilise_piece_prediction(contour_boxes, colour_boxes, contour_classes, colour_classes, 
                               piece_colours, previous_preds=None, previous_sources=None, 
                               previous_boxes=None):
    """Optimized stabilization with better caching"""
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
        # Validate box type
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
        
        # Decision logic
        # Option 1: Valid current colour prediction
        if best_overlap > 0 and best_colour_pred and hsv_piece_colour:
            pred_piece_colour = best_colour_pred.split('_')[0].lower()
            
            if pred_piece_colour == hsv_piece_colour:
                final_boxes.append(contour_box)
                final_preds.append(best_colour_pred)
                final_sources.append('colour')
                continue
        
        # Option 2: Hold previous colour prediction
        if prev_data and prev_data['source'] == 'colour':
            final_boxes.append(contour_box)
            final_preds.append(prev_data['pred'])
            final_sources.append('colour')
            continue
        
        # Option 3: Use contour prediction
        final_pred = f"{hsv_piece_colour}_{contour_pred}" if hsv_piece_colour else contour_pred
        final_boxes.append(contour_box)
        final_preds.append(final_pred)
        final_sources.append('contour')
    
    return final_boxes, final_preds, final_sources

def visualise_detections(img, boxes, predictions, bottom_loc):
    """Optimized visualization with pre-defined colors"""
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

def transform_boxes_remove_margin(boxes, margin_x=63, margin_y=70):
    """Optimized box transformation with pre-calculated values"""
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

# Initial board state
initial_state = [
    [Piece("castle", "white"), Piece("knight", "white"), Piece("bishop", "white"), Piece("queen", "white"), 
     Piece("king", "white"), Piece("bishop", "white"), Piece("knight", "white"), Piece("castle", "white")],
    [Piece("pawn", "white")] * 8,
    [None] * 8,
    [None] * 8,
    [None] * 8,
    [None] * 8,
    [Piece("pawn", "black")] * 8,
    [Piece("castle", "black"), Piece("knight", "black"), Piece("bishop", "black"), Piece("queen", "black"), 
     Piece("king", "black"), Piece("bishop", "black"), Piece("knight", "black"), Piece("castle", "black")]
]

board_state = initial_state

# Initialize detection system
if not init_detection_system():
    exit()

# Main loop variables
running = True
pts_src_buffer = []
MARGIN = 80
final_preds = None
final_sources = None
final_boxes = None
prev_board_state = board_state

# Performance optimization: only update visualizations when needed
UPDATE_INTERVAL = 1  # Update every N frames for less critical windows
frame_count = 0

while running:
    # print("current_state", current_state)
    frame_count += 1
    
    # Get current frame
    warp_margined, warp_unmargined, contoured_img, pts_src = get_current_frame()
    
    # Update stability buffer
    pts_src_buffer.append(pts_src)
    if len(pts_src_buffer) > 10:
        pts_src_buffer.pop(0)
    
    is_stable, max_std, details = calculate_corner_variation(pts_src_buffer, threshold=10.0)
    
    if warp_margined is not None:
        # Always show main warped images
        cv2.imshow('warp_unmargined', warp_unmargined)
        
        # Detect pieces
        contour_boxes, colour_boxes, contour_classes, colour_classes, \
        contour_annotated, colour_annotated, piece_colours = detect_pieces(warp_margined)
        
        # Update less critical visualizations less frequently
        if frame_count % UPDATE_INTERVAL == 0:
            cv2.imshow('warp_margined', warp_margined)
            cv2.imshow('contour_annotated', contour_annotated)
            cv2.imshow('colour_annotated', colour_annotated)
            
            h, w = colour_annotated.shape[:2]
            contour_cropped = contour_annotated[MARGIN:h-MARGIN, MARGIN:w-MARGIN]
            colour_cropped = colour_annotated[MARGIN:h-MARGIN, MARGIN:w-MARGIN]
            
            cv2.imshow('contour_cropped', contour_cropped)
            cv2.imshow('colour_cropped', colour_cropped)
        
        # Stabilize predictions
        final_boxes, final_preds, final_sources = stabilise_piece_prediction(
            contour_boxes, colour_boxes, contour_classes, colour_classes, 
            piece_colours, final_preds, final_sources, final_boxes
        )
        
        # Transform boxes and visualize
        final_boxes_unmargined, bottom_loc = transform_boxes_remove_margin(final_boxes)
        annotated_img_unmargined = visualise_detections(warp_unmargined, final_boxes_unmargined, 
                                                         final_preds, bottom_loc)
        cv2.imshow('annotated_img_unmargined', cv2.cvtColor(annotated_img_unmargined, cv2.COLOR_BGR2RGB))
    
    # Handle quit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False
    
    # Update GUI
    running = gui.do_gui(board_state, prev_board_state, current_state)
    
    # State machine logic
    if current_state == IDLE:
        if gui.setup_mode:
            prev_state = current_state
            current_state = STATIC
            print("Board setup - transitioning to STATIC")
    
    elif current_state == STATIC:
        # Continuously collect predictions when stable
        if is_stable and warp_unmargined is not None and final_preds is not None:
            current_board = get_cell_location(warp_unmargined, final_preds, bottom_loc)
            if current_board is not None:
                prediction_history.append(current_board)
                
                # Update board state with most common prediction every second
                if len(prediction_history) > 0:
                    board_state = get_most_common_board_state(prediction_history)
                    if board_state is not None:
                        prev_board_state = board_state
        
        # Detect movement
        if not is_stable:
            current_state = MOVING
            stability_start_time = None
            print("Movement detected - transitioning to MOVING")
    
    elif current_state == MOVING:
        # Clear prediction history when movement starts
        prediction_history.clear()
        voting_start_time = time.time()
        
        # Start stability timer when first detecting stability
        if is_stable:
            if stability_start_time is None:
                stability_start_time = time.time()
                print("Board stable, waiting 2 seconds before prediction...")
            
            # Check if we've been stable for the required delay
            elapsed_stability_time = time.time() - stability_start_time
            if elapsed_stability_time >= STABILITY_DELAY:
                current_state = PREDICT
                stability_start_time = None  # Reset for next time
                print("Stability delay complete - transitioning to PREDICT")
        else:
            # Reset timer if movement detected again
            stability_start_time = None
    
    elif current_state == PREDICT:
        # Collect predictions for voting period
        if voting_start_time is None:
            voting_start_time = time.time()

        elapsed_time = time.time() - voting_start_time

        # Collect predictions while stable
        if is_stable and warp_unmargined is not None and final_preds is not None:
            current_board = get_cell_location(warp_unmargined, final_preds, bottom_loc)
            if current_board is not None:
                prediction_history.append(current_board)

        # After voting period, determine final board state
        if elapsed_time >= VOTING_PERIOD:
            most_common_board = get_most_common_board_state(prediction_history)
            
            if most_common_board is not None:
                # Detect the move
                move_info = detect_move(prev_board_state, most_common_board)
                
                if move_info['valid']:
                    if move_info['is_legal']:
                        # Legal move detected
                        from_notation = get_chess_notation(move_info['from'][0], move_info['from'][1])
                        to_notation = get_chess_notation(move_info['to'][0], move_info['to'][1])
                        
                        piece_name = f"{move_info['piece'].colour} {move_info['piece'].type}"
                        
                        if move_info['captured']:
                            captured_name = f"{move_info['captured'].colour} {move_info['captured'].type}"
                            print(f"Legal move: {piece_name} from {from_notation} to {to_notation} (captured {captured_name})")
                        else:
                            print(f"Legal move: {piece_name} from {from_notation} to {to_notation}")
                        
                        # Highlight move as valid (GREEN)
                        gui.set_last_move(move_info['from'], move_info['to'], is_valid=True)
                        
                        # Update board state
                        board_state = most_common_board
                        prev_board_state = board_state
                        
                        # Check if either king is in check
                        check_status = check_if_in_check(board_state)

                        if check_status['white_in_check']:
                            white_king_pos = find_king(board_state, "white")
                            is_mate, attackers = check_if_checkmate(board_state, "white")
                            gui.set_check_status("white", white_king_pos, attackers, checkmate=is_mate)

                        elif check_status['black_in_check']:
                            black_king_pos = find_king(board_state, "black")
                            is_mate, attackers = check_if_checkmate(board_state, "black")
                            gui.set_check_status("black", black_king_pos, attackers, checkmate=is_mate)

                        else:
                            gui.clear_check_status()
                        
                        # Clear invalid move info since move was legal
                        invalid_move_info = None
                    else:
                        # Illegal move detected
                        print(f"Illegal move detected: {move_info['error']}")
                        print("Keeping previous board state - waiting for piece to be moved to valid position")
                        
                        # Store invalid move info
                        invalid_move_info = move_info
                        
                        # Highlight ONLY the destination square as RED (illegal position)
                        gui.set_last_move(None, move_info['to'], is_valid=False)
                        
                        # Transition to INVALID state
                        current_state = INVALID
                        print("Transitioning to INVALID state - piece must be moved to valid position")
                        
                        # Don't update board_state - keep previous valid state
                        # But update the tracking board to the current state
                        board_state = most_common_board
                else:
                    # Invalid move pattern
                    print(f"Invalid move pattern: {move_info['error']}")
                    print("Keeping previous board state")
                    
                    # Clear highlights for invalid pattern
                    gui.clear_last_move()
            else:
                print("No valid predictions collected, keeping previous state")
            
            # Reset voting (only if not transitioning to INVALID)
            if current_state != INVALID:
                prediction_history.clear()
                voting_start_time = None
                
                # Transition back to static
                current_state = STATIC
                print("Returning to STATIC state")
            else:
                # Still clear prediction history for INVALID state
                prediction_history.clear()
                voting_start_time = None
    
    elif current_state == INVALID:
        # Stay in INVALID state until piece is moved from the illegal square
        # Keep showing the red highlight
        
        # Continuously collect predictions when stable
        if is_stable and warp_unmargined is not None and final_preds is not None:
            current_board = get_cell_location(warp_unmargined, final_preds, bottom_loc)
            if current_board is not None:
                prediction_history.append(current_board)
                
                # Check if piece has been moved from invalid square
                if len(prediction_history) > 30:  # ~1 second of predictions
                    most_common_board = get_most_common_board_state(prediction_history)
                    
                    if most_common_board is not None and invalid_move_info is not None:
                        invalid_square = invalid_move_info['to']
                        
                        # Check if piece moved from invalid square
                        if check_piece_moved_from_invalid_square(board_state, most_common_board, invalid_square):
                            print("Piece moved from invalid square - checking new position...")
                            
                            # Detect the new move
                            move_info = detect_move(prev_board_state, most_common_board)
                            
                            if move_info['valid'] and move_info['is_legal']:
                                # Now it's a legal move!
                                from_notation = get_chess_notation(move_info['from'][0], move_info['from'][1])
                                to_notation = get_chess_notation(move_info['to'][0], move_info['to'][1])
                                
                                piece_name = f"{move_info['piece'].colour} {move_info['piece'].type}"
                                
                                if move_info['captured']:
                                    captured_name = f"{move_info['captured'].colour} {move_info['captured'].type}"
                                    print(f"Legal move: {piece_name} from {from_notation} to {to_notation} (captured {captured_name})")
                                else:
                                    print(f"Legal move: {piece_name} from {from_notation} to {to_notation}")
                                
                                # Highlight as valid (GREEN)
                                gui.set_last_move(move_info['from'], move_info['to'], is_valid=True)
                                
                                # Update board state
                                board_state = most_common_board
                                prev_board_state = board_state
                                
                                # Check if either king is in check
                                check_status = check_if_in_check(board_state)
                                
                                # Clear invalid move info
                                invalid_move_info = None
                                
                                # Return to STATIC state
                                prediction_history.clear()
                                current_state = STATIC
                                print("Valid move made - returning to STATIC state")
                            else:
                                # Still illegal - update the red highlight to new position
                                if move_info['valid']:
                                    print(f"Still illegal: {move_info['error']}")
                                    gui.set_last_move(None, move_info['to'], is_valid=False)
                                    invalid_move_info = move_info
                                    board_state = most_common_board
                                
                                prediction_history.clear()
        
        # Detect movement
        if not is_stable:
            print("Movement detected in INVALID state - waiting for stability...")
            prediction_history.clear()

# Cleanup
cleanup_camera()
gui.cleanup()