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

def crop_rows(warp_img, margin = 80):
    h, w = warp_img.shape[:2]
    unmargined_img = warp_img[0:h-margin, :]

    bottom_row = unmargined_img[h-190:, :]
    second_bottom_row = unmargined_img[h-280:h-140, :]
    return bottom_row, second_bottom_row
    # return unmargined_img


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
        
        # Detect pieces
        detection_result, boxes = chess_detection.detect_pieces(warped)
        # bottom_row_detection_result = chess_detection.detect_pieces(bottom_row)
        # second_bottom_row_detection_result = chess_detection.detect_pieces(second_bottom_row)

        # Visualize detections
        if detection_result is not None:
            annotated = detection_result.plot()
            cv2.imshow('Chess Piece Detection', annotated)
        # if bottom_row_detection_result is not None:
        #     annotated = bottom_row_detection_result.plot()
        #     cv2.imshow('bottom Piece Detection', annotated)
        # if second_bottom_row_detection_result is not None:
        #     annotated = second_bottom_row_detection_result.plot()
        #     cv2.imshow('second bottom Piece Detection', annotated)
    
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
        
        # Example transition (you can customize this)
        # if board_is_stable_for_2_seconds():
        #     current_state = PREDICT
    
    elif current_state == PREDICT:
        # Use detection results to predict move
        if detection_result is not None:
            # TODO: Convert detection_result to board state
            # detection_result.boxes contains all detected pieces
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