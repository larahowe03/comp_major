import cv2
import numpy as np
import matplotlib.pyplot as plt

# ------------------------
# Helpers
# ------------------------
def order_corners(pts):
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]
    top_right = pts[np.argmin(diff)]
    bottom_left = pts[np.argmax(diff)]
    return np.float32([top_left, top_right, bottom_right, bottom_left])

def get_square_name(row, col):
    file = chr(ord('a') + col)   # a–h
    rank = str(8 - row)          # ranks 8→1
    return file + rank

# ------------------------
# Stage 1: Detect and warp chessboard
# ------------------------
def detect_board(img, board_size=800):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)

    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    hull = cv2.convexHull(approx)
    if len(hull) > 4:
        hull = cv2.approxPolyDP(hull, epsilon, True)

    pts_src = np.float32([pt[0] for pt in hull])
    pts_src = order_corners(pts_src)

    pts_dst = np.float32([[0,0], [board_size,0], [board_size,board_size], [0,board_size]])
    M = cv2.getPerspectiveTransform(pts_src, pts_dst)
    warp = cv2.warpPerspective(img_rgb, M, (board_size, board_size))

    return warp

# ------------------------
# Stage 2: Piece detection
# ------------------------
def preprocess_mask(warp):
    """Build binary mask of potential pieces with grid lines removed."""
    gray = cv2.cvtColor(warp, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    edges = cv2.Canny(blur, 80, 180)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    mask = cv2.dilate(edges, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Remove vertical + horizontal grid lines
    # vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
    # horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    # temp1 = cv2.morphologyEx(mask, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
    # temp2 = cv2.morphologyEx(mask, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    # grid_lines = cv2.bitwise_or(temp1, temp2)
    # mask_no_lines = cv2.bitwise_and(mask, cv2.bitwise_not(grid_lines))

    # return mask_no_lines

def detect_pieces(mask, board_size=800, min_area=300):
    square_size = board_size // 8
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = {}
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue

        x,y,w,h = cv2.boundingRect(c)
        if h/w < 1.2:
            continue

        contour_poly = cv2.approxPolyDP(c, 3, True)
        bottom_point = tuple(c[c[:,:,1].argmax()][0])
        col = int(bottom_point[0] // square_size)
        row = int(bottom_point[1] // square_size)

        if 0 <= col < 8 and 0 <= row < 8:
            square = get_square_name(row, col)
            if square not in detections or area > detections[square][1]:
                detections[square] = (bottom_point, area, contour_poly)

    return detections

# ------------------------
# Main pipeline
# ------------------------
def process_chess_image(path):
    img = cv2.imread(path)
    warp = detect_board(img)

    # Build mask once
    mask_no_lines = preprocess_mask(warp)

    # Detect pieces
    detections = detect_pieces(mask_no_lines, board_size=warp.shape[0])

    # Visualization
    vis = warp.copy()
    for sq, (pt, area, contour_poly) in detections.items():
        cv2.drawContours(vis, [contour_poly], -1, (0,255,0), 2)
        cv2.circle(vis, pt, 6, (0,0,255), -1)
        cv2.putText(vis, sq, (pt[0]-20, pt[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1); plt.title("Piece mask"); plt.imshow(mask_no_lines, cmap='gray'); plt.axis("off")
    plt.subplot(1,2,2); plt.title("Detected pieces"); plt.imshow(vis); plt.axis("off")
    plt.show()

# ------------------------
# Run
# ------------------------
process_chess_image("img.jpg")
