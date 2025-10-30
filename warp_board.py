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
    
    if len(contours) == 0:
        print("No contours found")
        return img, img
    
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Take the largest contour
    largest_contour = contours[0]

    # Create mask for this contour
    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, [largest_contour], -1, 255, -1)

    # Get edges within the contour
    contour_edges = cv2.bitwise_and(edges, edges, mask=mask)

    # Detect straight lines within this contour
    lines = cv2.HoughLinesP(contour_edges, 1, np.pi/180, threshold=50,
                            minLineLength=80, maxLineGap=15)

    pts_src = None
    
    if lines is None:
        print("No lines found in contour, using contour approximation fallback")
    else:
        # Separate horizontal and vertical lines
        horizontal_lines = []
        vertical_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            angle = np.abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            # Filter short lines
            if length < 50:
                continue
            
            # Horizontal (angle close to 0° or 180°)
            if angle < 20 or angle > 160:
                horizontal_lines.append((x1, y1, x2, y2))
            # Vertical (angle close to 90°)
            elif 70 < angle < 110:
                vertical_lines.append((x1, y1, x2, y2))
        
        print(f"Found {len(horizontal_lines)} horizontal and {len(vertical_lines)} vertical lines")
        
        # Need at least 2 of each to find corners
        if len(horizontal_lines) >= 2 and len(vertical_lines) >= 2:
            # Find outermost lines
            top_line = min(horizontal_lines, key=lambda l: (l[1] + l[3]) / 2)
            bottom_line = max(horizontal_lines, key=lambda l: (l[1] + l[3]) / 2)
            left_line = min(vertical_lines, key=lambda l: (l[0] + l[2]) / 2)
            right_line = max(vertical_lines, key=lambda l: (l[0] + l[2]) / 2)
            
            # Function to find line intersection
            def line_intersection(line1, line2):
                x1, y1, x2, y2 = line1
                x3, y3, x4, y4 = line2
                
                denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
                if abs(denom) < 1e-10:
                    return None
                
                px = ((x1*y2 - y1*x2) * (x3 - x4) - (x1 - x2) * (x3*y4 - y3*x4)) / denom
                py = ((x1*y2 - y1*x2) * (y3 - y4) - (y1 - y2) * (x3*y4 - y3*x4)) / denom
                
                return (int(px), int(py))
            
            # Find 4 corner intersections
            top_left = line_intersection(top_line, left_line)
            top_right = line_intersection(top_line, right_line)
            bottom_left = line_intersection(bottom_line, left_line)
            bottom_right = line_intersection(bottom_line, right_line)
            
            corners = [top_left, top_right, bottom_right, bottom_left]
            
            # Check if all intersections were found
            if None not in corners:
                pts_src = np.float32(corners)
                print("✅ Found corners from line intersections")
                
                # Visualize
                blah = img.copy()
                cv2.drawContours(blah, [largest_contour], -1, (128, 128, 128), 2)
                
                # Draw detected lines
                cv2.line(blah, (top_line[0], top_line[1]), (top_line[2], top_line[3]), (0, 255, 0), 2)
                cv2.line(blah, (bottom_line[0], bottom_line[1]), (bottom_line[2], bottom_line[3]), (0, 0, 255), 2)
                cv2.line(blah, (left_line[0], left_line[1]), (left_line[2], left_line[3]), (255, 0, 0), 2)
                cv2.line(blah, (right_line[0], right_line[1]), (right_line[2], right_line[3]), (255, 255, 0), 2)
                
                # Draw corners
                for i, corner in enumerate(corners):
                    cv2.circle(blah, corner, 10, (255, 0, 255), -1)
                    cv2.putText(blah, str(i), (corner[0]-5, corner[1]+5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                print("Could not find all 4 corner intersections")

    # # Fallback: use contour approximation if line method failed
    # if pts_src is None:
    #     print("Using contour approximation fallback...")
    #     for c in contours:
    #         peri = cv2.arcLength(c, True)
    #         area = cv2.contourArea(c)
            
    #         if area < 1000:
    #             continue
            
    #         circularity = 4 * np.pi * area / (peri * peri)
            
    #         if circularity > 0.9:
    #             continue
            
    #         for eps in [0.01, 0.02, 0.03, 0.05]:
    #             approx = cv2.approxPolyDP(c, eps * peri, True)
                
    #             if len(approx) == 4:
    #                 pts = np.float32([pt[0] for pt in approx])
                    
    #                 angles = []
    #                 for i in range(4):
    #                     p1 = pts[i]
    #                     p2 = pts[(i + 1) % 4]
    #                     p3 = pts[(i + 2) % 4]
                        
    #                     v1 = p1 - p2
    #                     v2 = p3 - p2
                        
    #                     angle = np.abs(np.degrees(np.arctan2(np.linalg.det([v1, v2]), np.dot(v1, v2))))
    #                     angles.append(angle)
                    
    #                 if all(70 < angle < 110 for angle in angles):
    #                     pts_src = order_corners(pts)
    #                     break
            
    #         if pts_src is not None:
    #             break
        
        if pts_src is not None:
            blah = img.copy()
            cv2.drawContours(blah, contours, 0, (0, 255, 0), 2)

    # Final fallback
    if pts_src is None:
        print("All methods failed")
        return None, None

    # Warp perspective
    pts_dst = np.float32([[0,0],[board_size,0],[board_size,board_size],[0,board_size]])
    M = cv2.getPerspectiveTransform(pts_src, pts_dst)
    warp = cv2.warpPerspective(img_rgb, M, (board_size, board_size))

    return warp, blah


def process_chess_image(img):
    try:
        img, blah = detect_board(img)
    except:
        blah = img
        pass
        # try:
        #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # except:
        #     pass
    return img, blah


