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
# Hough Transform Functions
# ------------------------
def line_intersection(line1, line2):
    """
    Find intersection point of two lines in rho-theta format.
    Returns (x, y) or None if lines are parallel.
    """
    rho1, theta1 = line1
    rho2, theta2 = line2
    
    # Convert to ax + by = c format
    a1 = np.cos(theta1)
    b1 = np.sin(theta1)
    c1 = rho1
    
    a2 = np.cos(theta2)
    b2 = np.sin(theta2)
    c2 = rho2
    
    # Solve system of equations
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-10:  # parallel lines
        return None
    
    x = (b2 * c1 - b1 * c2) / det
    y = (a1 * c2 - a2 * c1) / det
    
    return (int(x), int(y))

def are_lines_similar(line1, line2, rho_threshold=20, theta_threshold=np.pi/18):
    """Check if two lines are similar (to merge duplicates)."""
    rho1, theta1 = line1
    rho2, theta2 = line2
    
    return (abs(rho1 - rho2) < rho_threshold and 
            abs(theta1 - theta2) < theta_threshold)

def merge_similar_lines(lines, rho_threshold=20, theta_threshold=np.pi/18):
    """Merge similar lines to remove duplicates."""
    if lines is None or len(lines) == 0:
        return []
    
    merged = []
    used = [False] * len(lines)
    
    for i, line1 in enumerate(lines):
        if used[i]:
            continue
        
        # Find all similar lines
        similar = [line1]
        for j, line2 in enumerate(lines[i+1:], i+1):
            if not used[j] and are_lines_similar(line1, line2, rho_threshold, theta_threshold):
                similar.append(line2)
                used[j] = True
        
        # Average the similar lines
        avg_rho = np.mean([l[0] for l in similar])
        avg_theta = np.mean([l[1] for l in similar])
        merged.append((avg_rho, avg_theta))
        used[i] = True
    
    return merged

def separate_lines(lines, theta_threshold=np.pi/4):
    """Separate lines into horizontal and vertical groups."""
    horizontal = []
    vertical = []
    
    for rho, theta in lines:
        # Vertical lines: theta near 0 or π
        if theta < theta_threshold or theta > np.pi - theta_threshold:
            vertical.append((rho, theta))
        # Horizontal lines: theta near π/2
        else:
            horizontal.append((rho, theta))
    
    return horizontal, vertical

def detect_grid_intersections(img, min_line_length=100, max_line_gap=10):
    """
    Use Hough Transform to detect grid lines and count intersections.
    Returns intersections and visualization data.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    
    # Hough Line Transform
    lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=150)
    
    if lines is None:
        return [], [], [], img.copy()
    
    # Convert to list of (rho, theta) tuples
    lines = [(line[0][0], line[0][1]) for line in lines]
    
    # Merge similar lines
    lines = merge_similar_lines(lines, rho_threshold=30, theta_threshold=np.pi/36)
    
    # Separate into horizontal and vertical
    h_lines, v_lines = separate_lines(lines)
    
    # Find all intersections
    intersections = []
    for h_line in h_lines:
        for v_line in v_lines:
            pt = line_intersection(h_line, v_line)
            if pt is not None:
                x, y = pt
                # Check if intersection is within image bounds
                if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                    intersections.append(pt)
    
    # Create visualization
    vis = img.copy()
    
    # Draw horizontal lines (blue)
    for rho, theta in h_lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 2000 * (-b))
        y1 = int(y0 + 2000 * (a))
        x2 = int(x0 - 2000 * (-b))
        y2 = int(y0 - 2000 * (a))
        cv2.line(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    # Draw vertical lines (green)
    for rho, theta in v_lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 2000 * (-b))
        y1 = int(y0 + 2000 * (a))
        x2 = int(x0 - 2000 * (-b))
        y2 = int(y0 - 2000 * (a))
        cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Draw intersections (red circles)
    for pt in intersections:
        cv2.circle(vis, pt, 5, (255, 0, 0), -1)
    
    return intersections, h_lines, v_lines, vis

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
def process_chess_image(img, show_hough=True):
    warp = detect_board(img)
    
    # Detect grid intersections using Hough Transform
    intersections, h_lines, v_lines, hough_vis = detect_grid_intersections(warp)
    print(f"Detected {len(h_lines)} horizontal lines")
    print(f"Detected {len(v_lines)} vertical lines")
    print(f"Found {len(intersections)} intersections")
    
    detections, mask, edges_no_lines, keep_mask = detect_pieces_edges(
        warp, min_area=200, angle_tol=12, min_gmag=30
    )

    vis = warp.copy()
    for sq, (pt, area, contour_poly) in detections.items():
        cv2.drawContours(vis, [contour_poly], -1, (0,255,0), 2)
        cv2.circle(vis, pt, 6, (0,0,255), -1)
        cv2.putText(vis, sq, (pt[0]-20, pt[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

    return warp, vis, hough_vis, intersections

# Example usage:
# img = cv2.imread('chessboard.jpg')
# warp, piece_vis, hough_vis, intersections = process_chess_image(img)
# 
# plt.figure(figsize=(18, 6))
# plt.subplot(1, 3, 1); plt.title("Warped Board"); plt.imshow(warp); plt.axis("off")
# plt.subplot(1, 3, 2); plt.title(f"Hough Lines ({len(intersections)} intersections)"); plt.imshow(hough_vis); plt.axis("off")
# plt.subplot(1, 3, 3); plt.title("Detected Pieces"); plt.imshow(piece_vis); plt.axis("off")
# plt.show()