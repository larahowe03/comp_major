import cv2
from collections import deque
import time
from allowable_moves import initial_state, check_if_checkmate, check_if_in_check
from camera_setup import get_current_frame, init_detection_system, cleanup_camera
from move_logic import detect_move, check_piece_moved_from_invalid_square
from chess_detection import detect_pieces
from process_detections import stabilise_piece_prediction, visualise_detections, transform_boxes_remove_margin
from board_stability import calculate_corner_variation
from board_functions import get_most_common_board_state, get_board_state, get_cell_on_board
import gui 

# ------------------------------------------------------------------------
# GLOBAL VARIABLES
# ------------------------------------------------------------------------

# State definitions
IDLE = 1
STATIC = 2
MOVING = 3
PREDICT = 4
INVALID = 5

current_state = IDLE
prev_state = IDLE

# Prediction voting system
VOTING_PERIOD = 1.0
STABILITY_DELAY = 2.0  
prediction_history = deque(maxlen=150)  # approx 5 seconds at 30fps
voting_start_time = None
stability_start_time = None

# Invalid move tracking - Stores invalid move details
invalid_move_info = None  

# Set intial board state
board_state = initial_state
prev_board_state = board_state

# Initialize detection system
if not init_detection_system():
    exit()


# ------------------------------------------------------------------------
# MAIN LOOP VARIABLES
# ------------------------------------------------------------------------

running = True
pts_src_buffer = []
MARGIN = 80
final_preds = None
final_sources = None
final_boxes = None

# Update visualisations every 1 second
UPDATE_INTERVAL = 1  
frame_count = 0


# ------------------------------------------------------------------------
# MAIN 
# ------------------------------------------------------------------------

while running:
    
    frame_count += 1
    
    # Get current frame
    warp_margined, warp_unmargined, contoured_img, pts_src = get_current_frame()
    
    # Update stability buffer
    pts_src_buffer.append(pts_src)
    if len(pts_src_buffer) > 10:
        pts_src_buffer.pop(0)
    
    is_stable, max_std, details = calculate_corner_variation(pts_src_buffer, threshold=10.0)
    
    if warp_margined is not None:
        
        # show main warped images
        cv2.imshow('warp_unmargined', warp_unmargined)
        
        # Detect pieces and bounding boxes from both models using the margined image
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
        
        # Stabilize predictions and get final predictions and boxes
        final_boxes, final_preds, final_sources = stabilise_piece_prediction(
            contour_boxes, colour_boxes, contour_classes, colour_classes, 
            piece_colours, final_preds, final_sources, final_boxes
        )
        
        # Transform boxes to unmargined image (just the board, no margin) and visualize
        final_boxes_unmargined, bottom_loc = transform_boxes_remove_margin(final_boxes)
        annotated_img_unmargined = visualise_detections(warp_unmargined, final_boxes_unmargined, 
                                                         final_preds, bottom_loc)
        cv2.imshow('annotated_img_unmargined', cv2.cvtColor(annotated_img_unmargined, cv2.COLOR_BGR2RGB))
    
    # Handle quit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False
    
    # Update GUI
    running = gui.do_gui(board_state, prev_board_state, current_state)
    
    # ---------------------------------------------------------------------
    # State machine logic
    # ---------------------------------------------------------------------
    
    if current_state == IDLE:
        if gui.setup_mode:
            prev_state = current_state
            current_state = STATIC
            print("Board setup - transitioning to STATIC")
    
    
    elif current_state == STATIC:
        # Continuously collect predictions when stable
        if is_stable and warp_unmargined is not None and final_preds is not None:
            current_board = get_board_state(warp_unmargined, final_preds, bottom_loc)
            
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
            current_board = get_board_state(warp_unmargined, final_preds, bottom_loc)
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
                        from_notation = get_cell_on_board(move_info['from'][0], move_info['from'][1])
                        to_notation = get_cell_on_board(move_info['to'][0], move_info['to'][1])
                        
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
                        check_status_white, white_king_pos = check_if_in_check(board_state, "white")
                        check_status_black, black_king_pos = check_if_in_check(board_state, "black")
                        
                        if check_status_white:
                            is_mate, attackers = check_if_checkmate(board_state, "white")
                            gui.set_check_status("white", white_king_pos, attackers, checkmate=is_mate)
                            
                        elif check_status_black:
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
            current_board = get_board_state(warp_unmargined, final_preds, bottom_loc)
            
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
                                from_notation = get_cell_on_board(move_info['from'][0], move_info['from'][1])
                                to_notation = get_cell_on_board(move_info['to'][0], move_info['to'][1])
                                
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
                                check_status_white, _ = check_if_in_check(board_state, "white")
                                check_status_black, _ = check_if_in_check(board_state, "black")
                                
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