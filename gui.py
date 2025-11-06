import pygame
import sys

# Initialize Pygame
pygame.init()

# Set up the window
WIDTH, HEIGHT = 800, 900  # Increased height for button area
BOARD_HEIGHT = 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Chess Board")

# Define colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BUTTON_COLOR = (70, 130, 180)  # Steel blue
BUTTON_HOVER_COLOR = (100, 149, 237)  # Cornflower blue
BUTTON_TEXT_COLOR = WHITE
GREEN_HIGHLIGHT = (0, 255, 0, 100)  # Valid move - transparent green
RED_HIGHLIGHT = (255, 0, 0, 100)    # Invalid move - transparent red

# Grid settings
ROWS = 8
COLS = 8
CELL_WIDTH = WIDTH // COLS
CELL_HEIGHT = BOARD_HEIGHT // ROWS

# Button settings
BUTTON_WIDTH = 200
BUTTON_HEIGHT = 60
BUTTON_X = (WIDTH - BUTTON_WIDTH) // 2
BUTTON_Y = BOARD_HEIGHT + 20

# Font
font = pygame.font.Font(None, 36)




# Setup mode flag
setup_mode = False

king_in_check_info = None 
attackers_info = []         # list of (row, col)
is_checkmate = False

def set_king_in_check(colour, pos):
    """Highlight the king's square and show text until cleared."""
    global king_in_check_info
    king_in_check_info = {'colour': colour, 'pos': pos}
    
def set_check_status(colour, king_pos, attackers, checkmate=False):
    """
    Display king in check/checkmate and highlight attackers.
    Args:
        colour: 'white' or 'black'
        king_pos: (row, col)
        attackers: list of (row, col) positions of pieces attacking the king
        checkmate: True if checkmate
    """
    global king_in_check_info, attackers_info, is_checkmate
    king_in_check_info = {'colour': colour, 'pos': king_pos}
    attackers_info = attackers or []
    is_checkmate = checkmate


def clear_check_status():
    """Clear any check/checkmate highlight."""
    global king_in_check_info, attackers_info, is_checkmate
    king_in_check_info = None
    attackers_info = []
    is_checkmate = False


def clear_king_in_check():
    """Remove king-in-check status."""
    global king_in_check_info
    king_in_check_info = None

# Move highlight info
last_move_info = None  # Will store {'from': (row, col), 'to': (row, col), 'valid': bool}

class Button:
    def __init__(self, x, y, width, height, text, color, hover_color, text_color):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.text_color = text_color
        self.is_hovered = False
        self.enabled = True
    
    def draw(self, surface):
        # Choose color based on enabled and hover state
        if not self.enabled:
            current_color = (128, 128, 128)  # Gray when disabled
        else:
            current_color = self.hover_color if self.is_hovered else self.color
        
        # Draw button rectangle
        pygame.draw.rect(surface, current_color, self.rect, border_radius=10)
        pygame.draw.rect(surface, BLACK, self.rect, 2, border_radius=10)  # Border
        
        # Draw text
        text_surface = font.render(self.text, True, self.text_color)
        text_rect = text_surface.get_rect(center=self.rect.center)
        surface.blit(text_surface, text_rect)
    
    def check_hover(self, mouse_pos):
        if self.enabled:
            self.is_hovered = self.rect.collidepoint(mouse_pos)
        else:
            self.is_hovered = False
    
    def is_clicked(self, mouse_pos):
        return self.enabled and self.rect.collidepoint(mouse_pos)
    
    def disable(self):
        self.enabled = False
    
    def enable(self):
        self.enabled = True


def draw_grid():
    """Draw a black and white checkerboard grid"""
    GRAY = (160, 160, 160)  # medium gray
    WHITE = (245, 245, 245) # soft white
    
    for row in range(ROWS):
        for col in range(COLS):
            # Alternate colors based on row and column
            if (row + col) % 2 == 0:
                color = GRAY
            else:
                color = WHITE
            
            # Calculate position
            x = col * CELL_WIDTH
            y = row * CELL_HEIGHT
            
            # Draw rectangle
            pygame.draw.rect(screen, color, (x, y, CELL_WIDTH, CELL_HEIGHT))


def draw_move_highlights():
    """Draw highlights for the last move (green for valid, red for invalid)"""
    global last_move_info
    
    if last_move_info is None:
        return
    
    from_pos = last_move_info.get('from')
    to_pos = last_move_info.get('to')
    is_valid = last_move_info.get('valid', False)
    
    # Choose color based on validity
    if is_valid:
        highlight_color = (0, 200, 0)  # Green for valid
        alpha = 120
    else:
        highlight_color = (200, 0, 0)  # Red for invalid
        alpha = 120
    
    # Create a transparent surface for highlights
    highlight_surface = pygame.Surface((CELL_WIDTH, CELL_HEIGHT))
    highlight_surface.set_alpha(alpha)
    highlight_surface.fill(highlight_color)
    
    # Draw "from" square
    if from_pos:
        row, col = from_pos
        x = col * CELL_WIDTH
        y = row * CELL_HEIGHT
        screen.blit(highlight_surface, (x, y))
        
        # Draw border
        pygame.draw.rect(screen, highlight_color, (x, y, CELL_WIDTH, CELL_HEIGHT), 4)
    
    # Draw "to" square
    if to_pos:
        row, col = to_pos
        x = col * CELL_WIDTH
        y = row * CELL_HEIGHT
        screen.blit(highlight_surface, (x, y))
        
        # Draw thicker border for destination
        pygame.draw.rect(screen, highlight_color, (x, y, CELL_WIDTH, CELL_HEIGHT), 6)

def draw_king_check_status():
    """Draw highlights for king in check or checkmate, plus attacker highlights."""
    global king_in_check_info, attackers_info, is_checkmate
    if king_in_check_info is None:
        return

    # --- Highlight the king’s square in red ---
    row, col = king_in_check_info['pos']
    x = col * CELL_WIDTH
    y = row * CELL_HEIGHT

    king_surface = pygame.Surface((CELL_WIDTH, CELL_HEIGHT))
    king_surface.set_alpha(160)
    king_surface.fill((255, 0, 0))  # Red for king
    screen.blit(king_surface, (x, y))
    pygame.draw.rect(screen, (255, 0, 0), (x, y, CELL_WIDTH, CELL_HEIGHT), 5)

    # --- Highlight attackers in blue ---
    for attacker in attackers_info:
        r, c = attacker
        ax = c * CELL_WIDTH
        ay = r * CELL_HEIGHT
        atk_surface = pygame.Surface((CELL_WIDTH, CELL_HEIGHT))
        atk_surface.set_alpha(140)
        atk_surface.fill((0, 0, 255))  # Blue for attacking pieces
        screen.blit(atk_surface, (ax, ay))
        pygame.draw.rect(screen, (0, 0, 255), (ax, ay, CELL_WIDTH, CELL_HEIGHT), 4)

    # --- Draw the text message ---
    if is_checkmate:
        msg = f"{king_in_check_info['colour'].capitalize()} King Checkmate!"
        color = (255, 0, 0)
    else:
        msg = f"{king_in_check_info['colour'].capitalize()} King in Check!"
        color = (200, 0, 0)

    text_surface = font.render(msg, True, color)
    text_rect = text_surface.get_rect(center=(WIDTH // 2, 20))
    screen.blit(text_surface, text_rect)


def draw_pieces(board):
    """Draw pieces on the board based on the initial state"""
    for row in range(ROWS):
        for col in range(COLS):
            piece = board[row][col]
            if piece is not None:
                # For simplicity, represent pieces as colored circles
                center_x = col * CELL_WIDTH + CELL_WIDTH // 2
                center_y = row * CELL_HEIGHT + CELL_HEIGHT // 2
                radius = min(CELL_WIDTH, CELL_HEIGHT) // 4
                
                try:
                    image = pygame.image.load("pieces/{}_{}.svg".format(piece.type, piece.colour))
                    image_rect = image.get_rect()
                    image_rect.center = (center_x, center_y)
                    screen.blit(image, image_rect)
                except:
                    pass


def handle_board_setup():
    """Handle the board setup button click"""
    global setup_mode
    setup_mode = True
    board_setup_button.disable()
    print("Board Setup mode activated!")


# Create button
board_setup_button = Button(
    BUTTON_X, BUTTON_Y, BUTTON_WIDTH, BUTTON_HEIGHT,
    "Board Setup", BUTTON_COLOR, BUTTON_HOVER_COLOR, BUTTON_TEXT_COLOR
)

# Clock for frame rate control
clock = pygame.time.Clock()
IDLE = 1
STATIC = 2
MOVING = 3
PREDICT = 4
CHANGED = 5
UPDATE_BOARD = 6


def set_last_move(from_pos, to_pos, is_valid):
    """
    Set the last move to be highlighted.
    
    Args:
        from_pos: (row, col) tuple for source square
        to_pos: (row, col) tuple for destination square
        is_valid: bool indicating if move was valid
    """
    global last_move_info
    last_move_info = {
        'from': from_pos,
        'to': to_pos,
        'valid': is_valid
    }


def clear_last_move():
    """Clear the move highlight."""
    global last_move_info
    last_move_info = None


def do_gui(board_state, prev_board_state, current_state):
    """
    Run one frame of the GUI.
    
    Args:
        board_state: Optional board state to display. If None, uses initial_state
    
    Returns:
        bool: False if window was closed, True otherwise
    """
    if current_state is MOVING:
        board_state = prev_board_state
    
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
            return False
        
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click
                mouse_pos = pygame.mouse.get_pos()
                if board_setup_button.is_clicked(mouse_pos):
                    handle_board_setup()
    
    # Get mouse position for hover effect
    mouse_pos = pygame.mouse.get_pos()
    board_setup_button.check_hover(mouse_pos)
    
    # Draw everything
    screen.fill((220, 220, 220))  # Light gray background
    
    # Draw the grid
    draw_grid()
    
    # Draw move highlights (before pieces so pieces are on top)
    draw_move_highlights()
    
    draw_king_check_status()
    
    # Draw pieces
    draw_pieces(board_state)
    
    # Draw button
    board_setup_button.draw(screen)
    
    # Update the display
    pygame.display.flip()
    
    # Control frame rate
    clock.tick(60)
    
    return True


def cleanup():
    """Clean up pygame resources"""
    pygame.quit()
    sys.exit()