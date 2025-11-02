import cv2
import numpy as np


def order_corners(pts):
    """Orders 4 corner points as [top-left, top-right, bottom-right, bottom-left]."""
    # Sort by y-coordinate (top to bottom)
    pts = pts[np.argsort(pts[:, 1])]
    
    # Top two points: sort by x (left to right)
    top_pts = pts[:2]
    top_pts = top_pts[np.argsort(top_pts[:, 0])]
    
    # Bottom two points: sort by x (left to right)
    bottom_pts = pts[2:]
    bottom_pts = bottom_pts[np.argsort(bottom_pts[:, 0])]
    
    # Combine: TL, TR, BR, BL
    return np.float32([top_pts[0], top_pts[1], bottom_pts[1], bottom_pts[0]])


def cluster_lines(lines, angle_threshold=10, distance_threshold=20):
    """Cluster similar lines together and return representative lines."""
    if lines is None or len(lines) == 0:
        return []
    
    clusters = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # Calculate line parameters
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        # Calculate midpoint and perpendicular distance from origin
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        
        # Find if this line belongs to existing cluster
        found_cluster = False
        for cluster in clusters:
            cluster_angle = cluster['angle']
            cluster_pos = cluster['position']
            
            # Check if angles are similar
            angle_diff = min(abs(angle - cluster_angle), 
                           abs(angle - cluster_angle + 180),
                           abs(angle - cluster_angle - 180))
            
            # Check if positions are similar (for parallel lines)
            pos_diff = np.sqrt((mid_x - cluster_pos[0])**2 + (mid_y - cluster_pos[1])**2)
            
            if angle_diff < angle_threshold and pos_diff < distance_threshold:
                # Add to this cluster
                cluster['lines'].append(line[0])
                cluster['lengths'].append(length)
                found_cluster = True
                break
        
        if not found_cluster:
            # Create new cluster
            clusters.append({
                'angle': angle,
                'position': (mid_x, mid_y),
                'lines': [line[0]],
                'lengths': [length]
            })
    
    # Get the longest line from each cluster
    representative_lines = []
    for cluster in clusters:
        # Find the longest line in the cluster
        max_idx = np.argmax(cluster['lengths'])
        representative_lines.append(cluster['lines'][max_idx])
    
    return representative_lines


def line_segment_intersection(line1, line2):
    """Find intersection of two line segments (extended to infinite lines)."""
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return None
    
    px = ((x1*y2 - y1*x2) * (x3 - x4) - (x1 - x2) * (x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2) * (y3 - y4) - (y1 - y2) * (x3*y4 - y3*x4)) / denom
    
    return (int(px), int(py))


def detect_board_hough(img, img_rgb, board_size=800):
    """
    Detect board using Hough lines - works even when corners are occluded by pieces.
    """
    print("\n🔍 Method 1: Hough Line Detection")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    
    # Detect line segments
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100,
                           minLineLength=100, maxLineGap=50)
    
    if lines is None or len(lines) < 4:
        print("  ❌ Not enough lines detected")
        return None, None
    
    print(f"  📊 Detected {len(lines)} raw lines")
    
    # Cluster similar lines
    clustered_lines = cluster_lines(lines, angle_threshold=15, distance_threshold=30)
    print(f"  📊 Clustered into {len(clustered_lines)} representative lines")
    
    # Separate into horizontal and vertical lines
    horizontal_lines = []
    vertical_lines = []
    
    for line in clustered_lines:
        x1, y1, x2, y2 = line
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        # Horizontal (angle close to 0° or 180°)
        if angle < 30 or angle > 150:
            horizontal_lines.append(line)
        # Vertical (angle close to 90°)
        elif 60 < angle < 120:
            vertical_lines.append(line)
    
    print(f"  📐 {len(horizontal_lines)} horizontal, {len(vertical_lines)} vertical lines")
    
    if len(horizontal_lines) < 2 or len(vertical_lines) < 2:
        print("  ❌ Not enough horizontal/vertical lines")
        return None, None
    
    # Find outermost lines
    top_line = min(horizontal_lines, key=lambda l: (l[1] + l[3]) / 2)
    bottom_line = max(horizontal_lines, key=lambda l: (l[1] + l[3]) / 2)
    left_line = min(vertical_lines, key=lambda l: (l[0] + l[2]) / 2)
    right_line = max(vertical_lines, key=lambda l: (l[0] + l[2]) / 2)
    
    # Find 4 corner intersections
    top_left = line_segment_intersection(top_line, left_line)
    top_right = line_segment_intersection(top_line, right_line)
    bottom_left = line_segment_intersection(bottom_line, left_line)
    bottom_right = line_segment_intersection(bottom_line, right_line)
    
    corners = [top_left, top_right, bottom_right, bottom_left]
    
    if None in corners:
        print("  ❌ Could not find all corner intersections")
        return None, None
    
    pts_src = np.float32(corners)
    
    # Visualize
    contoured_img = img.copy()
    
    # Draw all detected lines (thin gray)
    for line in clustered_lines:
        x1, y1, x2, y2 = line
        cv2.line(contoured_img, (x1, y1), (x2, y2), (128, 128, 128), 1)
    
    # Draw the 4 outermost lines (thick colored)
    cv2.line(contoured_img, tuple(top_line[:2]), tuple(top_line[2:]), (0, 255, 0), 3)
    cv2.line(contoured_img, tuple(bottom_line[:2]), tuple(bottom_line[2:]), (0, 255, 0), 3)
    cv2.line(contoured_img, tuple(left_line[:2]), tuple(left_line[2:]), (255, 0, 0), 3)
    cv2.line(contoured_img, tuple(right_line[:2]), tuple(right_line[2:]), (255, 0, 0), 3)
    
    # Draw corners
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    labels = ['TL', 'TR', 'BR', 'BL']
    for corner, color, label in zip(corners, colors, labels):
        cv2.circle(contoured_img, corner, 12, color, -1)
        cv2.circle(contoured_img, corner, 15, (255, 255, 255), 2)
        cv2.putText(contoured_img, label, (corner[0] - 15, corner[1] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    print("  ✅ Successfully detected board via Hough lines!")
    return pts_src, contoured_img


def detect_board_contour(img, img_rgb, board_size=800):
    """
    Detect board using contour approximation - works when full board outline is visible.
    """
    print("\n🔍 Method 2: Contour Detection")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    
    # Canny edges
    edges = cv2.Canny(blur, 30, 150)
    
    # Combine
    combined = cv2.bitwise_or(edges, cv2.bitwise_not(thresh))
    
    # Morphological operations
    kernel = np.ones((5, 5), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    combined = cv2.dilate(combined, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        print("  ❌ No contours found")
        return None, None
    
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # Try top 3 largest contours
    for idx, contour in enumerate(contours[:3]):
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        print(f"  📦 Contour {idx}: area={area:.0f}, perimeter={perimeter:.0f}")
        
        if area < 10000:
            continue
        
        # Try different approximations
        for epsilon_factor in [0.02, 0.03, 0.04, 0.05, 0.01]:
            epsilon = epsilon_factor * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) == 4 and cv2.isContourConvex(approx):
                corners = approx.reshape(4, 2).astype(np.float32)
                ordered_corners = order_corners(corners)
                
                # Check aspect ratio
                width1 = np.linalg.norm(ordered_corners[0] - ordered_corners[1])
                width2 = np.linalg.norm(ordered_corners[2] - ordered_corners[3])
                height1 = np.linalg.norm(ordered_corners[0] - ordered_corners[3])
                height2 = np.linalg.norm(ordered_corners[1] - ordered_corners[2])
                
                avg_width = (width1 + width2) / 2
                avg_height = (height1 + height2) / 2
                aspect_ratio = max(avg_width, avg_height) / min(avg_width, avg_height)
                
                if aspect_ratio < 2.5:
                    # Visualize
                    contoured_img = img.copy()
                    cv2.drawContours(contoured_img, [contour], -1, (0, 255, 0), 3)
                    
                    # Draw corners
                    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
                    labels = ['TL', 'TR', 'BR', 'BL']
                    for corner, color, label in zip(ordered_corners, colors, labels):
                        pt = tuple(corner.astype(int))
                        cv2.circle(contoured_img, pt, 12, color, -1)
                        cv2.circle(contoured_img, pt, 15, (255, 255, 255), 2)
                        cv2.putText(contoured_img, label, (pt[0] - 15, pt[1] - 20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    print(f"  ✅ Found valid quadrilateral (aspect ratio: {aspect_ratio:.2f})")
                    return ordered_corners, contoured_img
    
    print("  ❌ No valid quadrilateral found")
    return None, None


def detect_board(img, board_size=800):
    """
    Hybrid detection: tries Hough lines first (better for occluded corners),
    then falls back to contour detection.
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Try Hough line detection first (best for occluded corners)
    pts_src, contoured_img = detect_board_hough(img, img_rgb, board_size)
    
    # Fall back to contour detection
    if pts_src is None:
        pts_src, contoured_img = detect_board_contour(img, img_rgb, board_size)
    
    if pts_src is None:
        print("\n❌ Both detection methods failed")
        return None, None
    
    # Add margin by expanding source points outward
    margin = 80
    center_x = np.mean(pts_src[:, 0])
    center_y = np.mean(pts_src[:, 1])
    
    expansion_ratio = 1 + (2 * margin / board_size)
    expanded_pts_src = []
    
    for pt in pts_src:
        dx = pt[0] - center_x
        dy = pt[1] - center_y
        new_x = center_x + dx * expansion_ratio
        new_y = center_y + dy * expansion_ratio
        expanded_pts_src.append([new_x, new_y])
    
    expanded_pts_src = np.float32(expanded_pts_src)
    print(f"\nAdded {margin}px margin")
    
    # Perform perspective warp
    pts_dst = np.float32([[0, 0], [board_size, 0], [board_size, board_size], [0, board_size]])
    M = cv2.getPerspectiveTransform(expanded_pts_src, pts_dst)
    warp = cv2.warpPerspective(img_rgb, M, (board_size, board_size))
    
    print("✅ Board successfully warped!\n")
    return warp, contoured_img


def process_chess_image(img):
    """Main processing function"""
    try:
        warped, contoured = detect_board(img)
        if warped is None:
            return img, img
        return warped, contoured
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return img, img