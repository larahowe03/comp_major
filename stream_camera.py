import cv2
import pickle
from find_piece_places import process_chess_image

def undistort(img, K, d):
    return cv2.undistort(img, K, d, None, K)

if __name__ == "__main__":
    # Connecting to phone stream
    phone_ip = "10.19.204.195"
    port = "4747"

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
        print("Can't read")
        exit()

    # Load calibration coefficients
    with open("calibration_coefficients.pkl", "rb") as f:
        data = pickle.load(f)

    K = data['camera_matrix']
    d = data['distortion_coeffs']

    # Show the outputs
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Connection lost")
            break
        
        frame_undistorted = undistort(frame, K, d)

        cv2.imshow('Undistorted', frame_undistorted)
        cv2.imshow('Distorted', frame)

        warp, vis, hough_vis, intersections = process_chess_image(frame_undistorted)
        
        if len(intersections) >= 49:
            # new_warp = warp.copy()
            # new_vis = vis.copy()
            cv2.imshow('warp', warp)
            cv2.imshow('vis', vis)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

