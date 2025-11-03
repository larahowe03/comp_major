import pygame
import sys
from allowable_moves import initial_state

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


def do_gui(board_state=None):
    """
    Run one frame of the GUI.
    
    Args:
        board_state: Optional board state to display. If None, uses initial_state
    
    Returns:
        bool: False if window was closed, True otherwise
    """
    if board_state is None:
        board_state = initial_state
    
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