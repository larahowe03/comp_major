import pygame
from allowable_moves import initial_state

# Initialize Pygame
pygame.init()

# Set up the window
WIDTH, HEIGHT = 800, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Black and White Grid")

# Define colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Grid settings
ROWS = 8
COLS = 8
CELL_WIDTH = WIDTH // COLS
CELL_HEIGHT = HEIGHT // ROWS

def draw_grid():
    for row in range(ROWS):
        for col in range(COLS):
            # Draw the black and white grid
            if (row + col) % 2 == 0:
                color = WHITE
            else:
                color = BLACK
            
            # Get the position
            x = col * CELL_WIDTH
            y = row * CELL_HEIGHT
            
            # Draw the cell
            pygame.draw.rect(screen, color, (x, y, CELL_WIDTH, CELL_HEIGHT))

def draw_pieces(board):
    for row in range(ROWS):
        for col in range(COLS):
            piece = board[row][col]
            if piece is not None:
                # Get the locations of the pieces
                center_x = col * CELL_WIDTH + CELL_WIDTH // 2
                center_y = row * CELL_HEIGHT + CELL_HEIGHT // 2
                
                # Draw the image on the screen determined by the board state
                image = pygame.image.load("pieces/{}_{}.svg".format(piece.type, piece.colour))
                image_rect = image.get_rect()
                image_rect.center = (center_x, center_y)
                screen.blit(image, image_rect)


if __name__ == "__main__":
    running = True
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Draw the grid
        draw_grid()

        # Draw the pieces
        draw_pieces(initial_state)
        
        # Update
        pygame.display.flip()

    pygame.quit()
