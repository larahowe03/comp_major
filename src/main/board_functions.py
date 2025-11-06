import numpy as np
from collections import Counter
from allowable_moves import Piece

# ------------------------------------------------------------------------
# BOARD FUNCTIONS
# Contains functions to get the board states
# ------------------------------------------------------------------------

def map_detections(pieces_with_positions):
    """
    Create board state in the right format
    """
    board = [[None for _ in range(8)] for _ in range(8)]
    
    # Piece type mapping
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
        class_name = piece['class_name'].lower().strip()

        # Determine color and piece type
        if "white" in class_name:
            color = "white"
            piece_type = class_name.replace("white_", "")
        elif "black" in class_name:
            color = "black"
            piece_type = class_name.replace("black_", "")
        else:
            continue

        # map to correct naming format
        piece_type = type_mapping.get(piece_type, piece_type)
        
        # store piece in newly created state
        board[row][col] = Piece(piece_type, color)

    return board


def get_most_common_board_state(prediction_history):
    """
    Extract the most common board state from prediction history.
    Returns None if no valid predictions exist.
    """
    if not prediction_history:
        return None
    
    # Count occurrences of each board state
    board_state_counts = Counter()
    
    for board_state in prediction_history:
        board_tuple = tuple(
            tuple((piece.type, piece.colour) if piece is not None else None for piece in row)
                    for row in board_state)
        board_state_counts[board_tuple] += 1
    
    # Get most common board state as a tuple
    if not board_state_counts:
        return None
    most_common, _ = board_state_counts.most_common(1)[0]
    
    # Convert back to board state format
    board_state = []
    for row_tuple in most_common:
        row = []
        for cell in row_tuple:
            if cell is None:
                row.append(None)
            else:
                piece_type, piece_colour = cell
                row.append(Piece(piece_type, piece_colour))
        board_state.append(row)
    
    return board_state


def get_board_state(unmargined_img, final_preds, bottom_loc):
    """
    Get board state from the predicted labels and the locations of the pieces
    on the board.
    """
    
    if unmargined_img is None or final_preds is None or bottom_loc is None:
        return None
    
    h, w = unmargined_img.shape[:2]
    
    # Calculate cell size
    cell_width = w / 8
    cell_height = h / 8
    
    # Pre-calculate row boundaries
    row_boundaries = np.array([(i + 1) * cell_height for i in range(8)])
    
    pieces_with_positions = []
    
    for pred, loc in zip(final_preds, bottom_loc):
        if loc is None or len(loc) != 2:
            continue
        
        x, y = loc
        
        # Check boundaries
        if not (-cell_width <= x < w + cell_width and -cell_height <= y < h + cell_height):
            continue
        
        # Determine column
        col = max(0, min(7, int(x / cell_width)))
        
        # Find closest row - stop flucuations 
        distances = np.abs(row_boundaries - y)
        row = int(np.argmin(distances))
        
        # add to list
        pieces_with_positions.append({
            'row': row,
            'col': col,
            'class_name': pred
        })
    
    # Convert to board state
    board_state = map_detections(pieces_with_positions)
    
    return board_state


def get_cell_on_board(row, col, flip_board=False):
    """
    Convert row/col to chess notation (eg: a1, b4, etc..).
    """
    if flip_board:
        row = 7 - row
        col = 7 - col
    
    files = 'abcdefgh'
    ranks = '87654321'
    
    return files[col] + ranks[row]