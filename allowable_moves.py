# This file contains the matrices for the moves that can be performed by the chess pieces

class Piece:
    unique_id = 0  # Class-level variable shared across all instances

    def __init__(self, type, colour):
        self.type = type
        self.colour = colour
        # If it is a pawn, then you need to keep track of whether the first move is allowed, so there is this variable
        self.first_move = type == "pawn"
        self.id = Piece.unique_id
        Piece.unique_id += 1

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

def detect_move(prev_board, current_board):
    """
    Detect which piece moved by comparing board states.
    
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
    
    # Find all differences
    pieces_removed = []  # (row, col, piece)
    pieces_added = []    # (row, col, piece)
    
    for row in range(8):
        for col in range(8):
            prev_piece = prev_board[row][col]
            curr_piece = current_board[row][col]
            
            prev_occupied = prev_piece is not None
            curr_occupied = curr_piece is not None
            
            # Piece removed
            if prev_occupied and not curr_occupied:
                pieces_removed.append((row, col, prev_piece))
            
            # Piece added (and wasn't there before)
            elif not prev_occupied and curr_occupied:
                pieces_added.append((row, col, curr_piece))
            
            # Piece changed (capture case)
            elif prev_occupied and curr_occupied:
                if prev_piece.id != curr_piece.id:
                    pieces_removed.append((row, col, prev_piece))
                    pieces_added.append((row, col, curr_piece))
    
    # Validate move pattern
    # Normal move: 1 removed, 1 added
    # Capture: 2 removed (one piece + captured piece), 1 added
    
    if len(pieces_added) == 1 and len(pieces_removed) == 1:
        # Normal move
        from_row, from_col, moved_piece = pieces_removed[0]
        to_row, to_col, arrived_piece = pieces_added[0]
        captured_piece = None
        
        # Verify it's the same piece (by type and color)
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
        # Capture move
        to_row, to_col, arrived_piece = pieces_added[0]
        
        # One of the removed pieces should be at the destination (captured)
        # The other should be the piece that moved
        captured_piece = None
        moved_piece = None
        from_row, from_col = None, None
        
        for r, c, piece in pieces_removed:
            if r == to_row and c == to_col:
                captured_piece = piece
            else:
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
        
        # Verify piece types match
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
    # Pass prev_board directly to get_allowable_move
    try:
        allowable_moves = get_allowable_move(prev_board, from_row, from_col)
        is_legal = [to_row, to_col] in allowable_moves
    except Exception as e:
        is_legal = False
        error = f'Error checking legality: {str(e)}'
    
    return {
        'valid': True,
        'from': (from_row, from_col),
        'to': (to_row, to_col),  # FIXED: was (row, to_col)
        'piece': moved_piece,
        'captured': captured_piece,
        'is_legal': is_legal,
        'error': None if is_legal else 'Illegal move'
    }


def get_allowable_move(board, row, col):
    """Wrapper function that passes board to allowable move calculations"""
    piece = board[row][col]

    if piece is None:
        return []

    multiplier = -1 if piece.colour == "black" else 1

    if piece.type == "pawn":
        return get_allowable_move_pawn(board, piece, row, col, multiplier)
    if piece.type == "castle":
        return get_allowable_move_castle(board, piece, row, col)
    if piece.type == "bishop":
        return get_allowable_move_bishop(board, piece, row, col)
    if piece.type == "queen":
        return (get_allowable_move_castle(board, piece, row, col) + 
                get_allowable_move_bishop(board, piece, row, col))
    if piece.type == "knight":
        return get_allowable_move_knight(board, piece, row, col)
    if piece.type == "king":
        return get_allowable_move_king(board, piece, row, col, multiplier)


def get_allowable_move_pawn(board, piece, row, col, multiplier):
    """Pawn movement with board parameter"""
    possible_moves = [
        [row+1*multiplier, col],
        [row+2*multiplier, col],
        [row+1*multiplier, col-1],
        [row+1*multiplier, col+1]
    ]

    allowable_moves = [[row, col]]

    r1, c1 = possible_moves[0]
    r2, c2 = possible_moves[1]
    if in_bounds(r1, c1) and board[r1][c1] is None:
        allowable_moves.append([r1, c1])
        
        if piece.first_move and in_bounds(r2, c2) and board[r2][c2] is None:
            allowable_moves.append([r2, c2])

    r, c = possible_moves[2]
    if in_bounds(r, c):
        if board[r][c] is not None and board[r][c].colour != piece.colour:
            allowable_moves.append(possible_moves[2])

    r, c = possible_moves[3]
    if in_bounds(r, c):
        if board[r][c] is not None and board[r][c].colour != piece.colour:
            allowable_moves.append(possible_moves[3])
    
    return allowable_moves


def get_allowable_move_castle(board, piece, row, col):
    """Castle/Rook movement with board parameter"""
    allowable_moves = [[row, col]]

    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    for dr, dc in directions:
        r = row + dr
        c = col + dc
        while in_bounds(r, c):
            target = board[r][c]
            if target is None:
                allowable_moves.append([r, c])
            elif target.colour != piece.colour:
                allowable_moves.append([r, c])
                break
            else:
                break
            r += dr
            c += dc

    return allowable_moves


def get_allowable_move_knight(board, piece, row, col):
    """Knight movement with board parameter"""
    allowable_moves = [[row, col]]

    possible_moves = [
        [1, 2], [1, -2], [-1, 2], [-1, -2],
        [2, 1], [2, -1], [-2, 1], [-2, -1]
    ]

    for dr, dc in possible_moves:
        r = row + dr
        c = col + dc
        if in_bounds(r, c):
            target = board[r][c]
            if target is None or target.colour != piece.colour:
                allowable_moves.append([r, c])

    return allowable_moves


def get_allowable_move_bishop(board, piece, row, col):
    """Bishop movement with board parameter"""
    allowable_moves = [[row, col]]

    directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]

    for dr, dc in directions:
        r = row + dr
        c = col + dc
        while in_bounds(r, c):
            target = board[r][c]
            if target is None:
                allowable_moves.append([r, c])
            elif target.colour != piece.colour:
                allowable_moves.append([r, c])
                break
            else:
                break
            r += dr
            c += dc

    return allowable_moves


def get_squares_attacked_by_opponent(board, colour):
    """Get all squares attacked by opponent with board parameter"""
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


def get_allowable_move_king(board, piece, row, col, multiplier):
    """King movement with board parameter"""
    allowable_moves = [[row, col]]

    directions = [
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, 1), (1, -1), (-1, 1), (-1, -1)
    ]

    not_allowable_moves = get_squares_attacked_by_opponent(board, piece.colour)

    for dr, dc in directions:
        r = row + dr
        c = col + dc
        if not in_bounds(r, c):
            continue
        target = board[r][c]
        if [r, c] in not_allowable_moves:
            continue
        if target is None:
            allowable_moves.append([r, c])
        elif target.colour != piece.colour:
            allowable_moves.append([r, c])
    
    return allowable_moves


def in_bounds(r, c):
    """Check if position is within board bounds"""
    return 0 <= r < 8 and 0 <= c < 8