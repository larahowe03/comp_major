# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# # ------------------------
# # Helpers
# # ------------------------
# def order_corners(pts):
#     """Orders 4 corner points as [top-left, top-right, bottom-right, bottom-left]."""
#     s = pts.sum(axis=1)
#     diff = np.diff(pts, axis=1)
#     top_left = pts[np.argmin(s)]
#     bottom_right = pts[np.argmax(s)]
#     top_right = pts[np.argmin(diff)]
#     bottom_left = pts[np.argmax(diff)]
#     return np.float32([top_left, top_right, bottom_right, bottom_left])

# def get_square_name(row, col):
#     file = chr(ord('a') + col)   # a–h
#     rank = str(8 - row)          # ranks 8→1
#     return file + rank


# # ------------------------
# # Stage 1: Detect and warp chessboard
# # ------------------------
# def detect_board(img, board_size=800):
#     """Detects chessboard outer contour and warps it to a top-down view."""
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(gray, (5,5), 0)
#     edges = cv2.Canny(blur, 50, 150)

#     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     contour = max(contours, key=cv2.contourArea)

#     # Approximate corners
#     epsilon = 0.02 * cv2.arcLength(contour, True)
#     approx = cv2.approxPolyDP(contour, epsilon, True)
#     hull = cv2.convexHull(approx)
#     if len(hull) > 4:
#         hull = cv2.approxPolyDP(hull, epsilon, True)

#     pts_src = np.float32([pt[0] for pt in hull])
#     pts_src = order_corners(pts_src)

#     pts_dst = np.float32([[0,0], [board_size,0], [board_size,board_size], [0,board_size]])
#     M = cv2.getPerspectiveTransform(pts_src, pts_dst)
#     warp = cv2.warpPerspective(img_rgb, M, (board_size, board_size))

#     return warp


# def detect_board_lines(gray):
#     """Detect straight chessboard grid lines using Hough transform."""
#     edges = cv2.Canny(gray, 50, 150, apertureSize=3)
#     H, W = gray.shape[:2]
#     min_len = int(0.6 * min(H, W))
#     # Probabilistic Hough transform is cleaner
#     lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=500,
#                             minLineLength=min_len, maxLineGap=20)
#     return lines


# def create_line_mask(gray, lines):
#     mask = np.zeros_like(gray)

#     if lines is not None:
#         for l in lines:
#             x1, y1, x2, y2 = l[0]
#             cv2.line(mask, (x1,y1), (x2,y2), 255, 2)

#     return mask


# # ------------------------
# # Stage 2: Detect pieces on warped board
# # ------------------------
# def detect_pieces(warp, line_mask, min_area=800):
#     """Finds piece contours and bottom-most points inside a warped board."""
#     board_size = warp.shape[0]
#     square_size = board_size // 8

#     gray = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(gray, (7,7), 0)
#     edges = cv2.Canny(blur, 60, 150)
    
#     if line_mask is not None:
#         edges = cv2.bitwise_and(edges, cv2.bitwise_not(line_mask))

#     # Strengthen shapes
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
#     mask = cv2.dilate(edges, kernel, iterations=1)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

#     # Remove straight grid lines
#     # vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
#     # horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
#     # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, vertical_kernel)
#     # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, horizontal_kernel)

#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     detections = {}
#     for c in contours:
#         area = cv2.contourArea(c)
#         if area < min_area:
#             continue

#         x,y,w,h = cv2.boundingRect(c)
#         if h/w < 1.2:  # reject flat shapes
#             continue

#         contour_poly = cv2.approxPolyDP(c, 7, False)

#         # bottom-most point
#         bottom_point = tuple(c[c[:,:,1].argmax()][0])
#         col = int(bottom_point[0] // square_size)
#         row = int(bottom_point[1] // square_size)

#         if 0 <= col < 8 and 0 <= row < 8:
#             square = get_square_name(row, col)
#             if square not in detections or area > detections[square][1]:
#                 detections[square] = (bottom_point, area, contour_poly)

#     return detections, mask


# def detect_pieces_color(warp, min_area=500):
#     """
#     Detects chess pieces on the warped board using HSV color segmentation.
#     Works by masking out light + dark pieces and filtering blobs.
#     """
#     board_size = warp.shape[0]
#     square_size = board_size // 8

#     # --- Convert to HSV (better separation of brightness and color) ---
#     hsv = cv2.cvtColor(warp, cv2.COLOR_RGB2HSV)

#     # Masks for dark (black pieces) and light (white pieces)
#     lower_black = np.array([0, 0, 0])
#     upper_black = np.array([180, 255, 80])
#     mask_black = cv2.inRange(hsv, lower_black, upper_black)

#     lower_white = np.array([0, 0, 120])
#     upper_white = np.array([180, 50, 255])
#     mask_white = cv2.inRange(hsv, lower_white, upper_white)

#     # Combine
#     mask = cv2.bitwise_or(mask_black, mask_white)

#     # --- Clean mask with morphology ---
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

#     # --- Find contours (pieces) ---
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     detections = {}
#     for c in contours:
#         area = cv2.contourArea(c)
#         if area < min_area:
#             continue

#         x, y, w, h = cv2.boundingRect(c)
#         contour_poly = cv2.approxPolyDP(c, 5, True)

#         # bottom-most point
#         bottom_point = tuple(c[c[:,:,1].argmax()][0])
#         col = int(bottom_point[0] // square_size)
#         row = int(bottom_point[1] // square_size)

#         if 0 <= col < 8 and 0 <= row < 8:
#             square = get_square_name(row, col)
#             if square not in detections or area > detections[square][1]:
#                 detections[square] = (bottom_point, area, contour_poly)

#     return detections, mask


# def detect_pieces_combined(warp, min_area=2):
#     """
#     Combines color-based segmentation and contour/edge filtering
#     for robust piece detection.
#     """
#     board_size = warp.shape[0]
#     square_size = board_size // 8

#     # --- Step 1: Color segmentation ---
#     hsv = cv2.cvtColor(warp, cv2.COLOR_RGB2HSV)
#     lab = cv2.cvtColor(warp, cv2.COLOR_RGB2LAB)
#     L, A, B = cv2.split(lab)

#     # Dark pieces (HSV black mask)
#     lower_black = np.array([0, 0, 0])
#     upper_black = np.array([180, 255, 80])
#     mask_black = cv2.inRange(hsv, lower_black, upper_black)

#     # Light pieces (LAB L-channel threshold)
#     _, mask_white = cv2.threshold(L, 180, 255, cv2.THRESH_BINARY)

#     # Combine masks
#     mask_color = cv2.bitwise_or(mask_black, mask_white)

#     # --- Step 2: Edge/contour refinement ---
#     gray = cv2.cvtColor(warp, cv2.COLOR_RGB2GRAY)
#     blur = cv2.GaussianBlur(gray, (3,3), 0)
#     edges = cv2.Canny(blur, 60, 150)

#     # Keep only areas where color mask AND edges agree
#     mask_refined = cv2.bitwise_and(mask_color, edges)

#     # Clean with morphology
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1,1))
#     mask_refined = cv2.morphologyEx(mask_refined, cv2.MORPH_CLOSE, kernel, iterations=1)
#     mask_refined = cv2.morphologyEx(mask_refined, cv2.MORPH_OPEN, kernel, iterations=1)

#     # --- Step 3: Contour detection ---
#     contours, _ = cv2.findContours(mask_refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     detections = {}
#     for c in contours:
#         area = cv2.contourArea(c)
#         if area < min_area:
#             continue

#         x,y,w,h = cv2.boundingRect(c)
#         contour_poly = cv2.approxPolyDP(c, 5, True)

#         # bottom-most point
#         bottom_point = tuple(c[c[:,:,1].argmax()][0])
#         col = int(bottom_point[0] // square_size)
#         row = int(bottom_point[1] // square_size)

#         if 0 <= col < 8 and 0 <= row < 8:
#             square = get_square_name(row, col)
#             if square not in detections or area > detections[square][1]:
#                 detections[square] = (bottom_point, area, contour_poly)

#     return detections, mask_refined



# # ------------------------
# # Main pipeline
# # ------------------------
# def process_chess_image(path):
#     img = cv2.imread(path)

#     # Step 1: Warp chessboard
#     warp = detect_board(img)
    
#     gray = cv2.cvtColor(warp, cv2.COLOR_RGB2GRAY)
#     blur = cv2.GaussianBlur(gray, (5,5), 0)
    
#     # # detect lines
#     lines = detect_board_lines(blur)
#     line_mask = create_line_mask(gray, lines)

#     # initial piece edges
#     edges = cv2.Canny(blur, 60, 150)
#     mask = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)

#     # remove the board lines
#     mask_no_lines = cv2.bitwise_and(mask, cv2.bitwise_not(line_mask))

#     detections, mask = detect_pieces_combined(warp)
    
#     # Visualize results
#     vis = warp.copy()
    
#     for sq, (pt, area, contour_poly) in detections.items():
#         cv2.drawContours(vis, [contour_poly], -1, (0,255,0), 2)  # contour
#         cv2.circle(vis, pt, 6, (0,0,255), -1)                    # bottom point
#         cv2.putText(vis, sq, (pt[0]-20, pt[1]-10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

#     # Show
#     plt.figure(figsize=(12,6))
#     plt.subplot(1,2,1); plt.title("Piece mask"); plt.imshow(mask, cmap='gray'); plt.axis("off")
#     plt.subplot(1,2,2); plt.title("Detected pieces"); plt.imshow(vis); plt.axis("off")
#     plt.show()


# # ------------------------
# # Run
# # ------------------------
# process_chess_image("img.jpg")







# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# # ------------------------
# # Helpers
# # ------------------------
# def order_corners(pts):
#     """Orders 4 corner points as [top-left, top-right, bottom-right, bottom-left]."""
#     s = pts.sum(axis=1)
#     diff = np.diff(pts, axis=1)
#     top_left = pts[np.argmin(s)]
#     bottom_right = pts[np.argmax(s)]
#     top_right = pts[np.argmin(diff)]
#     bottom_left = pts[np.argmax(diff)]
#     return np.float32([top_left, top_right, bottom_right, bottom_left])

# def get_square_name(row, col):
#     file = chr(ord('a') + col)   # a–h
#     rank = str(8 - row)          # ranks 8→1
#     return file + rank


# # ------------------------
# # Stage 1: Detect and warp chessboard
# # ------------------------
# def detect_board(img, board_size=800):
#     """Detects chessboard outer contour and warps it to a top-down view."""
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(gray, (5,5), 0)
#     edges = cv2.Canny(blur, 50, 150)

#     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     contour = max(contours, key=cv2.contourArea)

#     # Approximate corners
#     epsilon = 0.02 * cv2.arcLength(contour, True)
#     approx = cv2.approxPolyDP(contour, epsilon, True)
#     hull = cv2.convexHull(approx)
#     if len(hull) > 4:
#         hull = cv2.approxPolyDP(hull, epsilon, True)

#     pts_src = np.float32([pt[0] for pt in hull])
#     pts_src = order_corners(pts_src)

#     pts_dst = np.float32([[0,0], [board_size,0], [board_size,board_size], [0,board_size]])
#     M = cv2.getPerspectiveTransform(pts_src, pts_dst)
#     warp = cv2.warpPerspective(img_rgb, M, (board_size, board_size))

#     return warp


# # ------------------------
# # Stage 2: Synthetic Grid Mask
# # ------------------------
# def generate_square_grid_mask(board_size=800, line_thickness=3, shift_x=50, shift_y=50):
#     """Generate a synthetic mask of the chessboard grid lines, with optional shift."""
#     mask = np.zeros((board_size, board_size), dtype=np.uint8)
#     step = board_size // 8

#     # Draw vertical lines (shifted by shift_x)
#     for x in range(0, board_size, step):
#         cv2.line(mask, (x+shift_x,0+shift_y), (x+shift_x,board_size+shift_y), 255, line_thickness)

#     # Draw horizontal lines (shifted by shift_y)
#     for y in range(0, board_size, step):
#         cv2.line(mask, (0+shift_x,y+shift_y), (board_size+shift_x,y+shift_y), 255, line_thickness)

#     return mask


# # ------------------------
# # Stage 3: Detect pieces (edges only, grid lines removed)
# # ------------------------
# def detect_pieces_edges(warp, min_area=200):
#     """Detects chess pieces using edges + contours, while masking out grid lines."""
#     board_size = warp.shape[0]
#     square_size = board_size // 8

#     # Edge detection
#     gray = cv2.cvtColor(warp, cv2.COLOR_RGB2GRAY)
#     blur = cv2.GaussianBlur(gray, (5,5), 0)
#     edges = cv2.Canny(blur, 60, 150)

#     # Generate synthetic chessboard grid mask
#     grid_mask = generate_square_grid_mask(board_size, line_thickness=3, shift_x=0, shift_y=-1)

#     # Remove grid lines from edge map
#     edges_no_lines = cv2.bitwise_and(edges, cv2.bitwise_not(grid_mask))

#     # Morphological cleanup
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
#     mask = cv2.dilate(edges_no_lines, kernel, iterations=2)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

#     # Contour detection
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     detections = {}
#     for c in contours:
#         area = cv2.contourArea(c)
#         if area < min_area:
#             continue

#         contour_poly = cv2.approxPolyDP(c, 5, True)

#         # bottom-most point
#         bottom_point = tuple(c[c[:,:,1].argmax()][0])
#         col = int(bottom_point[0] // square_size)
#         row = int(bottom_point[1] // square_size)

#         if 0 <= col < 8 and 0 <= row < 8:
#             square = get_square_name(row, col)
#             if square not in detections or area > detections[square][1]:
#                 detections[square] = (bottom_point, area, contour_poly)

#     return detections, mask, grid_mask


# # ------------------------
# # Main pipeline
# # ------------------------
# def process_chess_image(path):
#     img = cv2.imread(path)

#     # Step 1: Warp chessboard
#     warp = detect_board(img)

#     # Step 2: Detect pieces (grid lines removed)
#     detections, mask, grid_mask = detect_pieces_edges(warp)

#     # Visualize results
#     vis = warp.copy()
#     for sq, (pt, area, contour_poly) in detections.items():
#         cv2.drawContours(vis, [contour_poly], -1, (0,255,0), 2)  # contour
#         cv2.circle(vis, pt, 6, (0,0,255), -1)                    # bottom point
#         cv2.putText(vis, sq, (pt[0]-20, pt[1]-10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

#     # Show
#     plt.figure(figsize=(18,6))
#     plt.subplot(1,3,1); plt.title("Grid Mask"); plt.imshow(grid_mask, cmap='gray'); plt.axis("off")
#     plt.subplot(1,3,2); plt.title("Edges (lines removed)"); plt.imshow(mask, cmap='gray'); plt.axis("off")
#     plt.subplot(1,3,3); plt.title("Detected pieces"); plt.imshow(vis); plt.axis("off")
#     plt.show()


# # ------------------------
# # Run
# # ------------------------
# process_chess_image("img.jpg")




# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# # ------------------------
# # Helpers
# # ------------------------
# def order_corners(pts):
#     """Orders 4 corner points as [top-left, top-right, bottom-right, bottom-left]."""
#     s = pts.sum(axis=1)
#     diff = np.diff(pts, axis=1)
#     top_left = pts[np.argmin(s)]
#     bottom_right = pts[np.argmax(s)]
#     top_right = pts[np.argmin(diff)]
#     bottom_left = pts[np.argmax(diff)]
#     return np.float32([top_left, top_right, bottom_right, bottom_left])

# def get_square_name(row, col):
#     file = chr(ord('a') + col)   # a–h
#     rank = str(8 - row)          # ranks 8→1
#     return file + rank


# # ------------------------
# # Stage 1: Detect and warp chessboard
# # ------------------------
# def detect_board(img, board_size=800):
#     """Detects chessboard outer contour and warps it to a top-down view."""
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(gray, (5,5), 0)
#     edges = cv2.Canny(blur, 50, 150)

#     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     contour = max(contours, key=cv2.contourArea)

#     # Approximate corners
#     epsilon = 0.02 * cv2.arcLength(contour, True)
#     approx = cv2.approxPolyDP(contour, epsilon, True)
#     hull = cv2.convexHull(approx)
#     if len(hull) > 4:
#         hull = cv2.approxPolyDP(hull, epsilon, True)

#     pts_src = np.float32([pt[0] for pt in hull])
#     pts_src = order_corners(pts_src)

#     pts_dst = np.float32([[0,0], [board_size,0], [board_size,board_size], [0,board_size]])
#     M = cv2.getPerspectiveTransform(pts_src, pts_dst)
#     warp = cv2.warpPerspective(img_rgb, M, (board_size, board_size))

#     return warp


# # ------------------------
# # Stage 2: Detect pieces (edges only, no color masking)
# # ------------------------
# def detect_pieces_edges(warp, min_area=200):
#     """Detects chess pieces using only edges and contours (preserves colors)."""
#     board_size = warp.shape[0]
#     square_size = board_size // 8

#     # Edge detection
#     gray = cv2.cvtColor(warp, cv2.COLOR_RGB2GRAY)
#     blur = cv2.GaussianBlur(gray, (5,5), 0)
#     edges = cv2.Canny(blur, 60, 150)

#     # Morphological cleanup
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
#     mask = cv2.dilate(edges, kernel, iterations=1)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

#     # Contour detection
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     detections = {}
#     for c in contours:
#         area = cv2.contourArea(c)
#         if area < min_area:
#             continue

#         contour_poly = cv2.approxPolyDP(c, 5, True)

#         # bottom-most point
#         bottom_point = tuple(c[c[:,:,1].argmax()][0])
#         col = int(bottom_point[0] // square_size)
#         row = int(bottom_point[1] // square_size)

#         if 0 <= col < 8 and 0 <= row < 8:
#             square = get_square_name(row, col)
#             if square not in detections or area > detections[square][1]:
#                 detections[square] = (bottom_point, area, contour_poly)

#     return detections, mask


# # ------------------------
# # Main pipeline
# # ------------------------
# def process_chess_image(path):
#     img = cv2.imread(path)

#     # Step 1: Warp chessboard
#     warp = detect_board(img)

#     # Step 2: Detect pieces (edges only)
#     detections, mask = detect_pieces_edges(warp)

#     # Visualize results
#     vis = warp.copy()
#     for sq, (pt, area, contour_poly) in detections.items():
#         cv2.drawContours(vis, [contour_poly], -1, (0,255,0), 2)  # contour
#         cv2.circle(vis, pt, 6, (0,0,255), -1)                    # bottom point
#         cv2.putText(vis, sq, (pt[0]-20, pt[1]-10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

#     # Show
#     plt.figure(figsize=(12,6))
#     plt.subplot(1,2,1); plt.title("Edges mask"); plt.imshow(mask, cmap='gray'); plt.axis("off")
#     plt.subplot(1,2,2); plt.title("Detected pieces"); plt.imshow(vis); plt.axis("off")
#     plt.show()


# # ------------------------
# # Run
# # ------------------------
# process_chess_image("img.jpg")



import cv2
import numpy as np
import matplotlib.pyplot as plt

# ------------------------
# Helpers
# ------------------------
def order_corners(pts):
    """Orders 4 corner points as [top-left, top-right, bottom-right, bottom-left]."""
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
    """Detects chessboard outer contour and warps it to a top-down view."""
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
# Orientation-based grid suppression
# ------------------------
def non_axis_aligned_mask(gray, angle_tol=12, min_gmag=30):
    """
    Keep edges that are NOT near 0°/90°; suppress axis-aligned edges (checker lines).
    Returns a 0/255 uint8 mask to AND with an edge map.
    """
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    ang = (np.rad2deg(np.arctan2(gy, gx)) + 180.0) % 180.0  # [0,180)

    # near 0° or 180° (horizontal) or near 90° (vertical)
    near_h = (np.abs(ang - 0.0) < angle_tol) | (np.abs(ang - 180.0) < angle_tol)
    near_v = (np.abs(ang - 90.0) < angle_tol)
    grid_like = (near_h | near_v) & (mag > float(min_gmag))

    mask_keep = np.where(grid_like, 0, 255).astype(np.uint8)
    return mask_keep

# ------------------------
# Stage 2: Detect pieces (edges only, lines suppressed)
# ------------------------
def detect_pieces_edges(warp, min_area=200, angle_tol=15, min_gmag=40):
    """
    Detect chess pieces using edges/contours, suppressing checkerboard lines.
    Combines orientation filtering + contour aspect ratio filtering.
    """
    board_size = warp.shape[0]
    square_size = board_size // 8

    gray = cv2.cvtColor(warp, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # 1) Edge map
    edges = cv2.Canny(blur, 60, 150)

    # 2) Suppress axis-aligned edges (0°/90°)
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    ang = (np.rad2deg(np.arctan2(gy, gx)) + 180.0) % 180.0

    near_h = (np.abs(ang - 0.0) < angle_tol) | (np.abs(ang - 180.0) < angle_tol)
    near_v = (np.abs(ang - 90.0) < angle_tol)
    grid_like = (near_h | near_v) & (mag > float(min_gmag))
    keep_mask = np.where(grid_like, 0, 255).astype(np.uint8)

    edges_no_lines = cv2.bitwise_and(edges, keep_mask)

    # 3) Morphology (blob cleanup)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    # mask = cv2.morphologyEx(edges_no_lines, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.dilate(edges_no_lines, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=4)
    

    # 4) Contour detection
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = {}
    clean_mask = np.zeros_like(mask)

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue

        x,y,w,h = cv2.boundingRect(c)
        aspect = max(w,h) / (min(w,h)+1e-5)

        # filter out long skinny lines (likely checkerboard edges)
        if aspect > 6 and area < 3000:
            continue

        cv2.drawContours(clean_mask, [c], -1, 255, -1)

        contour_poly = cv2.approxPolyDP(c, 5, True)
        bottom_point = tuple(c[c[:,:,1].argmax()][0])
        col = int(bottom_point[0] // square_size)
        row = int(bottom_point[1] // square_size)

        if 0 <= col < 8 and 0 <= row < 8:
            square = get_square_name(row, col)
            if square not in detections or area > detections[square][1]:
                detections[square] = (bottom_point, area, contour_poly)

    return detections, clean_mask, edges_no_lines, keep_mask

# ------------------------
# Main pipeline
# ------------------------
def process_chess_image(path):
    img = cv2.imread(path)
    warp = detect_board(img)

    detections, mask, edges_no_lines, keep_mask = detect_pieces_edges(
        warp, min_area=200, angle_tol=12, min_gmag=30
    )

    vis = warp.copy()
    for sq, (pt, area, contour_poly) in detections.items():
        cv2.drawContours(vis, [contour_poly], -1, (0,255,0), 2)
        cv2.circle(vis, pt, 6, (0,0,255), -1)
        cv2.putText(vis, sq, (pt[0]-20, pt[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

    plt.figure(figsize=(18,6))
    plt.subplot(1,3,1); plt.title("Keep mask (non-axis edges)"); plt.imshow(keep_mask, cmap='gray'); plt.axis("off")
    plt.subplot(1,3,2); plt.title("Edges (lines suppressed)"); plt.imshow(mask, cmap='gray'); plt.axis("off")
    plt.subplot(1,3,3); plt.title("Detected pieces"); plt.imshow(vis); plt.axis("off")
    plt.show()

# ------------------------
# Run
# ------------------------
process_chess_image("img.jpg")
