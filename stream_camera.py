import cv2
from warp_board import process_chess_image

def calibrate_board(img):
    grid_size = (8, 6)  # (horizontal, vertical) number of corners on board

    # Convert to grayscale for detection
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect and refine corners
    ret, corners = cv2.findChessboardCorners(img_gray, grid_size, None)
    
    if ret and corners is not None and len(corners) > 0:
        print("Detected corners")
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        window_search = (11, 11)
        corners2 = cv2.cornerSubPix(img_gray, corners, window_search, (-1, -1), criteria)

        # Draw on the original color image
        cv2.drawChessboardCorners(img, grid_size, corners2, ret)

    return img

if __name__ == "__main__":
    # Replace with YOUR phone's IP and port from the DroidCam app
    phone_ip = "10.19.204.143"
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

        frame_drawn = calibrate_board(frame)
        processed = process_chess_image(frame)
        
        if ret:
            cv2.imshow('Phone Camera', frame_drawn)
            cv2.imshow('Processed', processed)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()