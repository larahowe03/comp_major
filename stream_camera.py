import cv2
import pickle
from warp_board import process_chess_image
from test_YOLOv8 import test_yolo
import matplotlib.pyplot as plt
from test_ResNet import resynet_test
import numpy as np

def clahethis(img):
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # Split into L, A, B channels
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    
    # Merge channels back
    lab_clahe = cv2.merge([l_clahe, a, b])
    
    # Convert back to BGR
    img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    
    return img_clahe

def undistort(img, K, d):
    return cv2.undistort(img, K, d, None, K)

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

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Connection lost")
            break

        frame_undistorted = undistort(frame, K, d)

        processed = process_chess_image(frame_undistorted)
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        
        clahed = clahethis(processed)

        if ret:
            cv2.imshow('Processed', processed)
            # cv2.imwrite("blah.png", processed)
            # yolod = resynet_test(processed)
        

        cv2.imshow('Undistorted', frame_undistorted)
        cv2.imshow('Distorted', frame)
        cv2.imshow('clahed', clahed)
        # cv2.imshow('thresh', thresh)
        # if yolod is not None and yolod.size > 0:
        #     cv2.imshow('yolod', yolod)
        # else:
        #     print("⚠️ Skipping frame — yolod is empty or invalid.")

        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

