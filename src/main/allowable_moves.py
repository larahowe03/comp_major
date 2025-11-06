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
    """
    Wrapper function that passes board to allowable move calculations
    """
    
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
    """
    Pawn movement with board parameter
    """
    
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
    """
    Castle/Rook movement with board parameter
    """
    
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
    """
    Knight movement with board parameter
    """
    
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
    """
    Bishop movement with board parameter
    """
    
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
    """
    Get all squares attacked by opponent with board parameter
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

def get_allowable_move_king(board, piece, row, col, multiplier):
    """
    King movement with board parameter
    """
    
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
    """
    Check if position is within board bounds
    """
    
    return 0 <= r < 8 and 0 <= c < 8