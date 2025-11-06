from enum import Enum
from allowable_moves import Piece
import gui
import chess_detection
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



def diff_board(prev_board, new_board):
    moved_from = None
    moved_to = None

    for r in range(8):
        for c in range(8):
            old_piece = prev_board[r][c]
            new_piece = new_board[r][c]
            if old_piece and not new_piece:
                moved_from = (r, c, old_piece)
            elif not old_piece and new_piece:
                moved_to = (r, c, new_piece)
    return moved_from, moved_to



def get_chess_notation(row, col, flip_board=False):
    """Convert row/col to chess notation (e.g., 'e4')."""
    if flip_board:
        row = 7 - row
        col = 7 - col
    
    files = 'abcdefgh'
    ranks = '87654321'  # Assuming row 0 is rank 8
    
    return files[col] + ranks[row]

def get_cell_location(unmargined_warp, boxes, margin=60, top_margin_extra=20, min_confidence=0.5):
      
    # Get unmargined board dimensions
    unmargined_height, unmargined_width = unmargined_warp.shape[:2]
    
    # Calculate cell size
    square_width = unmargined_width / 8
    square_height = unmargined_height / 8
    
    # Map each piece to board position
    pieces_with_positions = []
    for box_info in boxes:
        
        # FILTER LOW CONFIDENCE
        if box_info['confidence'] < min_confidence:
            continue
        
        x1, y1, x2, y2 = box_info['box']
        bottom_x = (x1 + x2) / 2
        bottom_y = y2
        
        # Adjust coordinates from margined space to unmargined space
        # Subtract left margin from x
        # Subtract top margin (margin + top_margin_extra) from y
        adjusted_x = bottom_x - margin
        adjusted_y = bottom_y - (margin + top_margin_extra)
        
        # Check if outside unmargined area
        if adjusted_x < 0 or adjusted_y < 0 or \
           adjusted_x > unmargined_width or adjusted_y > unmargined_height:
            continue  # Skip pieces outside the playable area
        
        # Calculate cell position in unmargined space
        col = int(adjusted_x / square_width)
        row = int(adjusted_y / square_height)
        
        # Clamp to valid range
        col = max(0, min(7, col))
        row = max(0, min(7, row))
        
        chess_pos = get_chess_notation(row, col)
        
        pieces_with_positions.append({
            **box_info,
            'row': row,
            'col': col,
            'position': chess_pos,
            'bottom_center': (bottom_x, bottom_y),  # Original margined coordinates
            'adjusted_bottom_center': (adjusted_x, adjusted_y)  # Unmargined coordinates
        })
    
    return pieces_with_positions

def crop_rows(warp_img, margin = 60):
    h, w = warp_img.shape[:2]
    unmargined_img = warp_img[margin+20:h-margin, margin:w-margin]

    bottom_row = unmargined_img[h-180:, :]
    second_bottom_row = unmargined_img[h-280:h-140, :]
    # return bottom_row, second_bottom_row
    return unmargined_img


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

# Initialize detection system
print("Initializing chess detection system...")
if not chess_detection.init_detection_system():
    print("Failed to initialize detection system")
    exit()

print("System initialized. Starting main loop...")

running = True
detection_result = None
warped_board = None

pts_src_buffer = []

MARGIN = 60
TOP_MARGIN_EXTRA = 20

while running:
    # Always update GUI
    running = gui.do_gui(board_state)
    
    # Get current frame from camera
    undistorted, warped, pts_src = chess_detection.get_current_frame()

    pts_src_buffer.append(pts_src)

    if len(pts_src_buffer) > 20:
        pts_src_buffer.pop(0)

    is_stable, max_std, details = calculate_corner_variation(pts_src_buffer, threshold=10.0)

    print("is_stable", is_stable)
    print("current_state", current_state)
    
    # Show camera feed if available
    if undistorted is not None:
        cv2.imshow('Camera Feed', undistorted)
    
    if warped is not None:
        warped_board = warped
        cv2.imshow('Warped Board', warped)
        # bottom_row, second_bottom_row = crop_rows(warped)
        # unmargined = crop_rows(warped)
        
        h, w = warped.shape[:2]
        unmargined = warped[MARGIN+TOP_MARGIN_EXTRA:h-MARGIN, MARGIN:w-MARGIN]
        
        # Detect pieces
        detection_result = chess_detection.detect_pieces(warped)
        
        # unmargined_result = chess_detection.detect_pieces(unmargined)
        # second_bottom_row_detection_result = chess_detection.detect_pieces(second_bottom_row)
        
        boxes_info = []
        for box in detection_result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            boxes_info.append({
                'box': [float(x1), float(y1), float(x2), float(y2)],
                'bottom': float(y2),
                'confidence': float(box.conf[0].cpu().numpy()),
                'class_id': int(box.cls[0].cpu().numpy()),
                'class_name': detection_result.names[int(box.cls[0].cpu().numpy())]
            })
            
        if boxes_info:
            pieces_with_positions = get_cell_location(
                unmargined,  # Only used for dimensions
                boxes_info,  # These are in margined coordinates
                margin=MARGIN, 
                top_margin_extra=TOP_MARGIN_EXTRA,
                min_confidence=0.6
            )
            
        if pieces_with_positions:
            new_board_state = detections_to_board(pieces_with_positions)
            moved_from, moved_to = diff_board(board_state, new_board_state)

            if moved_from and moved_to:
                r1, c1, piece = moved_from
                r2, c2, _ = moved_to
                move_str = f"{piece.colour} {piece.type}: {get_chess_notation(r1, c1)} → {get_chess_notation(r2, c2)}"
                print("♟ Detected move:", move_str)

            # Replace the global board state
            board_state = new_board_state
            
            # for piece in pieces_with_positions:
            #     print(f"  {piece['class_name']} at {piece['position']} (conf: {piece['confidence']:.2f})")
        

        # Visualize detections
        if detection_result is not None:
            annotated = detection_result.plot()
            cv2.imshow('Margined Frame', annotated)
    
    # Check for quit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False
    
    # State machine logic
    if current_state == IDLE:
        # Wait for board setup button
        if gui.setup_mode:
            prev_state = current_state
            current_state = STATIC
            print("Game beginning - transitioning to STATIC state")
    
    elif current_state == STATIC:
        # Monitor board for changes
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
print("Cleaning up...")
chess_detection.cleanup_camera()
gui.cleanup()