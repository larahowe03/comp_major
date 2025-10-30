import cv2
import pickle
import os
from datetime import datetime

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

    with open("calibration_coefficients.pkl", "rb") as f:
        data = pickle.load(f)

    K = data['camera_matrix']
    d = data['distortion_coeffs']

    # Create output directory if it doesn't exist
    output_dir = "captured_frames"
    os.makedirs(output_dir, exist_ok=True)

    print("\n=== Controls ===")
    print("SPACE: Capture frame")
    print("Q: Quit")
    print("================\n")

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Undistort the frame (optional)
        h, w = frame.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(K, d, (w, h), 1, (w, h))
        undistorted = cv2.undistort(frame, K, d, None, new_camera_matrix)

        # Display the frame
        cv2.imshow('DroidCam Feed - Press SPACE to capture, Q to quit', undistorted)

        key = cv2.waitKey(1) & 0xFF

        # Press SPACE to capture
        if key == ord(' '):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{output_dir}/frame_{frame_count:04d}_{timestamp}.jpg"
            cv2.imwrite(filename, undistorted)
            frame_count += 1
            print(f"Captured: {filename}")

        # Press Q to quit
        elif key == ord('q'):
            print(f"\nTotal frames captured: {frame_count}")
            break

    cap.release()
    cv2.destroyAllWindows()