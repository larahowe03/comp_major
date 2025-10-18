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


# this board variable will come from the detector, this example just flows through some things
board = None

# This is the initial state of the board
initial_state = [
    [Piece("castle", "white"), Piece("knight", "white"), Piece("bishop", "white"), Piece("queen", "white"), Piece("king", "white"), Piece("bishop", "white"), Piece("knight", "white"), Piece("castle", "white")],
    [Piece("pawn", "white"), Piece("pawn", "white"), Piece("pawn", "white"), Piece("pawn", "white"), Piece("pawn", "white"), Piece("pawn", "white"), Piece("pawn", "white"), Piece("pawn", "white")],
    [None, None, None, None, None, None, None, None],
    [None, None, None, None, None, None, None, None],
    [None, None, None, None, None, None, None, None],
    [None, None, None, None, None, None, None, None],
    [Piece("pawn", "black"), Piece("pawn", "black"), Piece("pawn", "black"), Piece("pawn", "black"), Piece("pawn", "black"), Piece("pawn", "black"), Piece("pawn", "black"), Piece("pawn", "black")],
    [Piece("castle", "black"), Piece("knight", "black"), Piece("bishop", "black"), Piece("queen", "black"), Piece("king", "black"), Piece("bishop", "black"), Piece("knight", "black"), Piece("castle", "black")]
]

# Testing now will just use initial state
board = initial_state

def in_bounds(r, c):
    return 0 <= r < 8 and 0 <= c < 8

### PAWN ###
# Returns locations that the pawn can go to
def get_allowable_move_pawn(piece, row, col, multiplier):
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
        if board[r][c] != None and board[r][c].colour != piece.colour:
            allowable_moves.append(possible_moves[2])

    r, c = possible_moves[3]
    if in_bounds(r, c):
        if board[r][c] != None and board[r][c].colour != piece.colour:
            allowable_moves.append(possible_moves[3])
    
    return allowable_moves

def get_allowable_move_castle(piece, row, col):
    allowable_moves = [[row, col]]

    directions = [
        (1, 0),   # down
        (-1, 0),  # up
        (0, 1),   # right
        (0, -1)   # left
    ]

    for dr, dc in directions:
        r = row + dr
        c = col + dc
        while in_bounds(r, c):
            target = board[r][c]
            if target is None:
                allowable_moves.append([r, c])
            elif target.colour != piece.colour:
                allowable_moves.append([r, c])
                break  # can't go beyond capture
            else:
                break  # blocked by own piece
            r += dr
            c += dc

    return allowable_moves

def get_allowable_move_knight(piece, row, col):
    allowable_moves = [[row, col]]

    possible_moves = [
        [1, 2], 
        [1, -2], 
        [-1, 2], 
        [-1, -2], 
        [2, 1], 
        [2, -1], 
        [-2, 1], 
        [-2, -1]
    ]

    for dr, dc in possible_moves:
        r = row + dr
        c = col + dc
        if in_bounds(r, c):
            target = board[r][c]
            if target is None or target.colour != piece.colour:
                allowable_moves.append([r, c])

    return allowable_moves

def get_allowable_move_bishop(piece, row, col):
    allowable_moves = [[row, col]]

    directions = [
        (1, 1), 
        (1, -1), 
        (-1, 1),   
        (-1, -1)   
    ]

    for dr, dc in directions:
        r = row + dr
        c = col + dc
        while in_bounds(r, c):
            target = board[r][c]
            if target is None:
                allowable_moves.append([r, c])
            elif target.colour != piece.colour:
                allowable_moves.append([r, c])
                break  # can't go beyond capture
            else:
                break  # blocked by own piece
            r += dr
            c += dc

    return allowable_moves

def get_squares_attacked_by_opponent(colour):
    not_allowable_moves = []

    for r in range(len(board)):
        for c in range(len(board[r])):
            if not in_bounds(r, c):
                continue
            target = board[r][c]
            if target is None or target.type == "king":
                continue
            if colour != target.colour:
                not_allowable_moves.extend(get_allowable_move(r, c))
    
    return not_allowable_moves


def get_allowable_move_king(piece, row, col, multiplier):
    allowable_moves = [[row, col]]

    directions = [
        (1, 0),   # down
        (-1, 0),  # up
        (0, 1),   # right
        (0, -1),   # left
        (1, 1), 
        (1, -1), 
        (-1, 1),   
        (-1, -1)   
    ]

    not_allowable_moves = get_squares_attacked_by_opponent(piece.colour)

    for dr, dc in directions:
        r = row + dr
        c = col + dc
        if not in_bounds(r, c):
            continue
        target = board[r][c]
        # Checking that the move does not put the king in check
        if [r, c] in not_allowable_moves:
            continue
        # Getting moves that are allowed
        if target is None:
            allowable_moves.append([r, c])
        elif target.colour != piece.colour:
            allowable_moves.append([r, c])
    
    return allowable_moves

def check_in_check(row, col):
    king = board[row][col]
    
    if [row, col] in get_squares_attacked_by_opponent(king.colour):
        return True
    else:
        return False


def get_allowable_move(row, col):
    piece = board[row][col]

    if piece is None:
        return []

    if piece.colour == "black":
        multiplier = -1
    if piece.colour == "white":
        multiplier = 1

    if piece.type == "pawn":
        return get_allowable_move_pawn(piece, row, col, multiplier)
    if piece.type == "castle":
        return get_allowable_move_castle(piece, row, col)
    if piece.type == "bishop":
        return get_allowable_move_bishop(piece, row, col)
    if piece.type == "queen":
        return get_allowable_move_castle(piece, row, col) + get_allowable_move_bishop(piece, row, col)
    if piece.type == "knight":
        return get_allowable_move_knight(piece, row, col)
    if piece.type == "king":
        return get_allowable_move_king(piece, row, col, multiplier)

def move_piece(new_row, new_col, initial_row, initial_col, allowable_move):
    if [new_row, new_col] not in allowable_move:
        print("Cannot move here")
    else:
        board[new_row][new_col] = board[initial_row][initial_col]
        board[initial_row][initial_col] = None
        if board[new_row][new_col].type == "pawn":
            board[new_row][new_col].first_move = False
