import numpy as np
import cv2
import matplotlib.pyplot as plt

# RIGHT NOW, THIS DOESN'T WORK, AS THE CHESSBOARD FUNCTION IS SPECIFIC AND DOES NOT HANDLE OCCLUSION


def get_outer_corners(inner_corners, grid_size):
    """
    Calculate outer corners of chessboard from inner corners.
    
    For a chessboard with grid_size (cols, rows) inner corners,
    we have (cols+1, rows+1) actual squares.
    We need to extrapolate outward by one square width in each direction.
    """
    grid_width, grid_height = grid_size
    corners = inner_corners.reshape(-1, 2)
    
    # Get the four corner internal points
    tl_inner = corners[0]  # Top-left internal corner
    tr_inner = corners[grid_width - 1]  # Top-right internal corner
    br_inner = corners[-1]  # Bottom-right internal corner
    bl_inner = corners[-grid_width]  # Bottom-left internal corner
    
    # Get adjacent corners to calculate spacing
    # Top-left corner: use the point to its right and below
    tl_right = corners[1]
    tl_below = corners[grid_width]
    
    # Top-right corner: use the point to its left and below
    tr_left = corners[grid_width - 2]
    tr_below = corners[2 * grid_width - 1]
    
    # Bottom-right corner: use the point to its left and above
    br_left = corners[-2]
    br_above = corners[-(grid_width + 1)]
    
    # Bottom-left corner: use the point to its right and above
    bl_right = corners[-grid_width + 1]
    bl_above = corners[-2 * grid_width]
    
    # Calculate spacing vectors (one square width/height)
    tl_horizontal = tl_inner - tl_right
    tl_vertical = tl_inner - tl_below
    
    tr_horizontal = tr_inner - tr_left
    tr_vertical = tr_inner - tr_below
    
    br_horizontal = br_inner - br_left
    br_vertical = br_inner - br_above
    
    bl_horizontal = bl_inner - bl_right
    bl_vertical = bl_inner - bl_above
    
    # Extrapolate outer corners (move one square outward)
    tl_outer = tl_inner + tl_horizontal + tl_vertical
    tr_outer = tr_inner + tr_horizontal + tr_vertical
    br_outer = br_inner + br_horizontal + br_vertical
    bl_outer = bl_inner + bl_horizontal + bl_vertical
    
    outer_corners = np.array([
        tl_outer,
        tr_outer,
        br_outer,
        bl_outer
    ], dtype=np.float32)
    
    return outer_corners

def calibrate_board(img):
    img2 = img.copy()
    grid_size = (7, 7)  # (horizontal, vertical) number of corners on board

    # Convert to grayscale for detection
    img_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("img_gray.png", img_gray)

    # Detect and refine corners
    ret, corners = cv2.findChessboardCorners(img_gray, grid_size, None)
    
    if ret and corners is not None and len(corners) > 0:
        print("Detected corners")
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        window_search = (11, 11)
        corners2 = cv2.cornerSubPix(img_gray, corners, window_search, (-1, -1), criteria)

        # Calculate outer corners
        outer_corners = get_outer_corners(corners2, grid_size)

        # Draw on the original color image
        cv2.drawChessboardCorners(img2, grid_size, corners2, ret)
        print(len(outer_corners))
        return img2, outer_corners
    return None, None

def warp_board(img, corners):
    src_points = corners.copy()

    width_final = 200
    height_final = 200

    dst_boints = np.array([
        [0, 0],
        [width_final, 0],
        [width_final, height_final],
        [0, height_final]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src_points, dst_boints)
    return cv2.warpPerspective(img, M, (width_final, height_final))


if __name__ == "__main__":
    # Replace with YOUR phone's IP and port from the DroidCam app
    phone_ip = "10.19.202.136"
    port = "4747"

    # DroidCam streaming URLs - try these in order:
    urls = [
        f"http://{phone_ip}:{port}/video",
        f"http://{phone_ip}:{port}/mjpegfeed",
    ]

    cap = None
    for url in urls:
        print(f"Trying: {url}")
        cap = cv2.VideoCapture(url)
        if cap.isOpened():
            print(f"Connected successfully to {url}")
            break
        cap.release()

    if not cap or not cap.isOpened():
        print("Could not connect. Check:")
        print("1. Phone and laptop on same WiFi")
        print("2. IP address is correct")
        print("3. DroidCam app is running")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Connection lost")
            break
        
        # frame_drawn, corners = calibrate_board(frame)
        frame_drawn, corners = detect_aruco_corners(frame)
        if frame_drawn is not None:
            warped_image = warp_board(frame, corners)
            
            cv2.imshow('Phone Camera', frame)
            cv2.imshow('Drawn', frame_drawn)
            cv2.imshow('Warped', warped_image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

