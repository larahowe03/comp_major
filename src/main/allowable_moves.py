# ------------------------------------------------------------------------
# ALLOWABLE MOVES
# This file contains the matrices for the moves that can be performaned
# by the chess pieces
# ------------------------------------------------------------------------

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


def get_allowable_move(board, row, col):
    piece = board[row][col]

    if piece is None:
        return []

    # Since black pieces move opposite direction to white pieces
    multiplier = 1
    if piece.colour == "black":
        multiplier = -1

    if piece.type == "pawn":
        return get_allowable_move_pawn(board, piece, row, col, multiplier)
    if piece.type == "castle":
        return get_allowable_move_castle(board, piece, row, col)
    if piece.type == "bishop":
        return get_allowable_move_bishop(board, piece, row, col)
    if piece.type == "queen":
        # Queen is essentially combination of two move types
        return (get_allowable_move_castle(board, piece, row, col) + 
                get_allowable_move_bishop(board, piece, row, col))
    if piece.type == "knight":
        return get_allowable_move_knight(board, piece, row, col)
    if piece.type == "king":
        return get_allowable_move_king(board, piece, row, col, multiplier)

# ALL the allowable moves functions essentially check all the allowed locations for all pieces 
# and check that this is within the 8x8 board and that there is not already another piece of the same colour in that location

def get_allowable_move_pawn(board, piece, row, col, multiplier):    
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

# This is so that you can check that the king can move, because it cant move anywhere that an opponent is located
def get_squares_attacked_by_opponent(board, colour):
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

# Finsd where the king is
def find_king(board, colour):
    if board is None:
        return None
    
    for row in range(8):
        for col in range(8):
            piece = board[row][col]
            if piece is not None and piece.type == "king" and piece.colour == colour:
                return (row, col)
    return None

# Checks if the king is in checkmate
def check_if_checkmate(board, colour):    
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

    # Check if the king is being attacked and king has no possible moves
    # If true then checkmate
    if list(king_pos) in attacked_squares:
        king_moves = get_allowable_move(board, king_pos[0], king_pos[1])
        safe_moves = [m for m in king_moves if m not in attacked_squares]
        if not safe_moves:
            return True, attackers

    return False, attackers

# Check if the specified king is in check.
def check_if_in_check(board, colour):    
    if board is None:
        return False, None

    king_pos = find_king(board, colour)
    if king_pos is None:
        return False, None

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

def in_bounds(row, col):    
    return 0 <= row < 8 and 0 <= col < 8
