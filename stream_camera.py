import cv2
<<<<<<< HEAD
from warp_board import process_chess_image
=======
import pickle
>>>>>>> 4a788cbef5b65221c2a886571995a054fbd66f39

def undistort(img, K, d):
    return cv2.undistort(img, K, d, None, K)

if __name__ == "__main__":
    # Replace with YOUR phone's IP and port from the DroidCam app
<<<<<<< HEAD
    phone_ip = "10.19.204.143"
=======
    phone_ip = "10.19.204.195"
>>>>>>> 4a788cbef5b65221c2a886571995a054fbd66f39
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

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Connection lost")
            break
<<<<<<< HEAD

        frame_drawn = calibrate_board(frame)
        processed = process_chess_image(frame)
        
        if ret:
            cv2.imshow('Phone Camera', frame_drawn)
            cv2.imshow('Processed', processed)
=======
        
        frame_undistorted = undistort(frame, K, d)

        cv2.imshow('Undistorted', frame_undistorted)
        cv2.imshow('Distorted', frame)
>>>>>>> 4a788cbef5b65221c2a886571995a054fbd66f39
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

