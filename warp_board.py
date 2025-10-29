import cv2
import numpy as np
import matplotlib.pyplot as plt

# ------------------------
# Helper: order corners
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

# ------------------------
# Helper: line intersection
# ------------------------
def line_intersection(line1, line2):
    rho1, theta1 = line1
    rho2, theta2 = line2
    a1, b1 = np.cos(theta1), np.sin(theta1)
    a2, b2 = np.cos(theta2), np.sin(theta2)
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-10:
        return None
    x = (b2 * rho1 - b1 * rho2) / det
    y = (a1 * rho2 - a2 * rho1) / det
    return (int(x), int(y))


def detect_board_hough(img, board_size=800, debug=False):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)

    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=200)
    if lines is None:
        return None, [], [], []

    horizontals, verticals = [], []
    for l in lines:
        rho, theta = l[0]
        if abs(theta - np.pi/2) < np.pi/6:   # vertical
            verticals.append((rho, theta))
        elif abs(theta) < np.pi/6 or abs(theta - np.pi) < np.pi/6:  # horizontal
            horizontals.append((rho, theta))

    if not horizontals or not verticals:
        return None, horizontals, verticals, []

    # extreme lines
    left = min(verticals, key=lambda l: l[0])
    right = max(verticals, key=lambda l: l[0])
    top = min(horizontals, key=lambda l: l[0])
    bottom = max(horizontals, key=lambda l: l[0])

    # intersections
    tl = line_intersection(top, left)
    tr = line_intersection(top, right)
    br = line_intersection(bottom, right)
    bl = line_intersection(bottom, left)
    intersections = [tl,tr,br,bl]

    if None in intersections:
        return None, horizontals, verticals, intersections

    # warp
    pts_src = order_corners(np.float32(intersections))
    pts_dst = np.float32([[0,0],[board_size,0],[board_size,board_size],[0,board_size]])
    M = cv2.getPerspectiveTransform(pts_src, pts_dst)
    warp = cv2.warpPerspective(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), M, (board_size, board_size))

    if debug:
        vis = img.copy()
        # draw lines
        for (rho,theta) in horizontals + verticals:
            a, b = np.cos(theta), np.sin(theta)
            x0, y0 = a*rho, b*rho
            x1 = int(x0 + 2000*(-b))
            y1 = int(y0 + 2000*(a))
            x2 = int(x0 - 2000*(-b))
            y2 = int(y0 - 2000*(a))
            cv2.line(vis, (x1,y1), (x2,y2), (0,255,0), 2)
            
        return warp, horizontals, verticals, intersections, vis

    return warp, horizontals, verticals, intersections, None

# ------------------------
# Main: hybrid detection
# ------------------------
def detect_board(img, board_size=800):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # Morphological close to connect edges
    kernel = np.ones((5,5), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    pts_src = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        for eps in [0.01, 0.02, 0.03, 0.05]:
            approx = cv2.approxPolyDP(c, eps * peri, True)
            if len(approx) == 4:
                pts_src = np.float32([pt[0] for pt in approx])
                pts_src = order_corners(pts_src)
                break
        if pts_src is not None:
            break

    # If contour failed, try Hough
    if pts_src is None:
        print("Contour failed, trying Hough fallback...")
        return detect_board_hough(img, board_size)

    # Warp from contour
    pts_dst = np.float32([[0,0],[board_size,0],[board_size,board_size],[0,board_size]])
    M = cv2.getPerspectiveTransform(pts_src, pts_dst)
    warp = cv2.warpPerspective(img_rgb, M, (board_size, board_size))
    return warp


def process_chess_image(path):
    img = cv2.imread(path)
    
    warp = detect_board(img)
    warp2 = detect_board(warp)
    
    return cv2.cvtColor(warp2, cv2.COLOR_BGR2RGB)



# warp = process_chess_image(img)
# warp = process_chess_image("img.png")
# plt.imshow(warp)
# plt.show()

