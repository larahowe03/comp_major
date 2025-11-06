from allowable_moves import in_bounds, get_allowable_move

# ------------------------------------------------------------------------
# MOVE LOGIC FUNCTIONS
# Contains functions related to moves made and validation
# ------------------------------------------------------------------------

def detect_move(prev_board, current_board):
    """
    Detect which piece moved by comparing board states.
    Only tracks: occupied -> empty (piece removed) and empty -> occupied (piece added)
    
    Returns:
        dict: {
            'valid': bool,
            'from': (row, col) or None,
            'to': (row, col) or None,
            'piece': Piece object or None,
            'captured': Piece object or None,
            'is_legal': bool,
            'error': str or None
        }
    """
    
    if prev_board is None or current_board is None:
        return {
            'valid': False,
            'from': None,
            'to': None,
            'piece': None,
            'captured': None,
            'is_legal': False,
            'error': 'Invalid board state'
        }
    
    # Find all differences - only track occupancy changes
    pieces_removed = []  # (row, col, piece) - was occupied, now empty
    pieces_added = []    # (row, col, piece) - was empty, now occupied
    
    for row in range(8):
        for col in range(8):
            prev_piece = prev_board[row][col]
            curr_piece = current_board[row][col]
            
            prev_occupied = prev_piece is not None
            curr_occupied = curr_piece is not None
            
            # Piece removed: was occupied, now empty
            if prev_occupied and not curr_occupied:
                pieces_removed.append((row, col, prev_piece))
            
            # Piece added: was empty, now occupied
            elif not prev_occupied and curr_occupied:
                pieces_added.append((row, col, curr_piece))
            
            # If both occupied or both empty, ignore (no change in occupancy)
    
    # Validate move pattern
    # Normal move: 1 removed, 1 added
    # Capture: 2 removed (piece moved away + captured piece disappeared), 1 added
    
    if len(pieces_added) == 1 and len(pieces_removed) == 1:
        # Normal move: one square emptied, one square filled
        from_row, from_col, moved_piece = pieces_removed[0]
        to_row, to_col, arrived_piece = pieces_added[0]
        captured_piece = None
        
        # Verify it's the same piece type and color (basic sanity check)
        if (moved_piece.type != arrived_piece.type or 
            moved_piece.colour != arrived_piece.colour):
            return {
                'valid': False,
                'from': None,
                'to': None,
                'piece': None,
                'captured': None,
                'is_legal': False,
                'error': 'Piece type/color mismatch'
            }
    
    elif len(pieces_added) == 1 and len(pieces_removed) == 2:
        # Capture move: two squares emptied, one square filled
        to_row, to_col, arrived_piece = pieces_added[0]
        
        # One of the removed pieces should be at the destination (captured)
        # The other should be the piece that moved
        captured_piece = None
        moved_piece = None
        from_row, from_col = None, None
        
        for r, c, piece in pieces_removed:
            if r == to_row and c == to_col:
                # This piece was at the destination - it was captured
                captured_piece = piece
            else:
                # This piece moved away from its square
                moved_piece = piece
                from_row, from_col = r, c
        
        if moved_piece is None or captured_piece is None:
            return {
                'valid': False,
                'from': None,
                'to': None,
                'piece': None,
                'captured': None,
                'is_legal': False,
                'error': 'Invalid capture pattern'
            }
        
        # Verify the moving piece matches the arrived piece
        if (moved_piece.type != arrived_piece.type or 
            moved_piece.colour != arrived_piece.colour):
            return {
                'valid': False,
                'from': None,
                'to': None,
                'piece': None,
                'captured': None,
                'is_legal': False,
                'error': 'Piece type/color mismatch in capture'
            }
    
    else:
        return {
            'valid': False,
            'from': None,
            'to': None,
            'piece': None,
            'captured': None,
            'is_legal': False,
            'error': f'Invalid move: {len(pieces_removed)} removed, {len(pieces_added)} added'
        }
    
    # Check if move is legal using allowable_moves
    try:
        allowable_moves = get_allowable_move(prev_board, from_row, from_col)
        is_legal = [to_row, to_col] in allowable_moves
    except Exception as e:
        is_legal = False
        error = f'Error checking legality: {str(e)}'
    
    return {
        'valid': True,
        'from': (from_row, from_col),
        'to': (to_row, to_col),
        'piece': moved_piece,
        'captured': captured_piece,
        'is_legal': is_legal,
        'error': None if is_legal else 'Illegal move'
    }
    
    
def check_piece_moved_from_invalid_square(prev_board, current_board, invalid_square):
    """
    Check if the piece on the invalid square has been moved
    """
    
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
    """
    Find the position of a king on the board
    """
    
    if board is None:
        return None
    
    for row in range(8):
        for col in range(8):
            piece = board[row][col]
            if piece is not None and piece.type == "king" and piece.colour == colour:
                return (row, col)
    
    return None


def get_squares_attacked_by_opponent(board, colour):
    """
    Get all squares attacked by opponent
    """
    
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
    Checks if the specified king is in checkmate
    """
    
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

    # Basic checkmate test
    # Check if the king is being attacked, and king has no possible moves. If true, then checkmate
    if list(king_pos) in attacked_squares:
        from allowable_moves import get_allowable_move
        king_moves = get_allowable_move(board, king_pos[0], king_pos[1])
        safe_moves = [m for m in king_moves if m not in attacked_squares]
        if not safe_moves:
            return True, attackers

    return False, attackers


def check_if_in_check(board, colour):
    """
    Check if the specified king is in check.
    """
    
    if board is None:
        return False, None, []

    king_pos = find_king(board, colour)
    if king_pos is None:
        return False, None, []

    attacked_squares = get_squares_attacked_by_opponent(board, colour)
    in_check = list(king_pos) in attacked_squares

    attackers = []
    if in_check:
        # Identify which opponent pieces are attacking the king
        for r in range(8):
            for c in range(8):
                piece = board[r][c]
                if piece is not None and piece.colour != colour:
                    moves = get_allowable_move(board, r, c)
                    if list(king_pos) in moves:
                        attackers.append((r, c))
        
        print(f"{colour.upper()} King is in check!")

    return in_check, king_pos