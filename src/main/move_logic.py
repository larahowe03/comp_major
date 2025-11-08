from allowable_moves import in_bounds, get_allowable_move

# ------------------------------------------------------------------------
# MOVE LOGIC FUNCTIONS
# Contains functions related to moves made and validation
# ------------------------------------------------------------------------

# Since the model always fluctuates piece predictions but is quite stable for piece locations,
# a move is only detected when a full location change happens in the board rather than when the board overall changes
def detect_move(prev_board, current_board):    
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
    
    # Find all differences in piece locations
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
                
    # A normal move is when 1 piece was removed ad one was addded
    # A piece is taken if two pieces are removed and one is added
    
    if len(pieces_added) == 1 and len(pieces_removed) == 1:
        # Normal move: one square emptied, one square filled
        from_row, from_col, moved_piece = pieces_removed[0]
        to_row, to_col, arrived_piece = pieces_added[0]
        captured_piece = None
        
        # Verify the same piece colour and type
        if (moved_piece.type != arrived_piece.type or moved_piece.colour != arrived_piece.colour):
            return {
                'valid': False,
                'from': None,
                'to': None,
                'piece': None,
                'captured': None,
                'is_legal': False,
                'error': 'Piece type/color mismatch'
            }
    
    # cjecking if a piece has been taken
    elif len(pieces_added) == 1 and len(pieces_removed) == 2:
        to_row, to_col, arrived_piece = pieces_added[0]
        
        # One of the removed pieces should be at the new location and the other should be gone
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
        
        # Verify the same piece colour and type
        if (moved_piece.type != arrived_piece.type or moved_piece.colour != arrived_piece.colour):
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
    allowable_moves = get_allowable_move(prev_board, from_row, from_col)
    is_legal = [to_row, to_col] in allowable_moves
    
    return {
        'valid': True,
        'from': (from_row, from_col),
        'to': (to_row, to_col),
        'piece': moved_piece,
        'captured': captured_piece,
        'is_legal': is_legal,
        'error': None if is_legal else 'Illegal move'
    }

# Function for while it is invalid checking if it fixes and moves to a valid location
def check_piece_moved_from_invalid_square(prev_board, current_board, invalid_square):    
    if prev_board is None or current_board is None or invalid_square is None:
        return False
    
    row, col = invalid_square
    prev_piece = prev_board[row][col]
    curr_piece = current_board[row][col]
    
    # If there was a piece and now there isn't, it moved
    if prev_piece is not None and curr_piece is None:
        return True
    
    return False