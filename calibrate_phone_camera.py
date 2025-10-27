import time
import pickle 
import cv2
import numpy as np
import os

def get_img_info(images, img_dir):
    corners_list = []
    pattern_points_list = []

    grid_size = (6, 6)

    for i, im in enumerate(images):
        if not im.endswith("png"):
            print(f"Skipping {im}")
            continue
        
        img = cv2.imread(os.path.join(img_dir, im), cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"Failed to load {im}")
            continue

        # get corners
        ret, corners = cv2.findChessboardCorners(img, grid_size, None)
        if not ret:
            print(f"Failed to find corners in {im}")
            continue

        print(f"Found corners in {im}")

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        window_search = (11, 11)
        corners2 = cv2.cornerSubPix(img, corners, window_search, (-1, -1), criteria)
        corners_list.append(corners2)

        # draw corners
        imgc = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        cv2.drawChessboardCorners(imgc, grid_size, corners2, ret)

        # Build numpy array containing (x,y,z) coordinates of corners, relative to board itself
        pattern_points = np.zeros((np.prod(grid_size), 3), np.float32)
        pattern_points[:, :2] = np.indices(grid_size).T.reshape(-1, 2) # fill-in X-Y points of grid
        pattern_points_list.append(pattern_points)

        h, w = img.shape[:2]
    
    if len(corners_list) == 0:
        print("No valid images found!")
        return None, None, None, None
    
    return corners_list, pattern_points_list, h, w

def capture_images():
    # Streaming
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
        print("Can't connect")
        exit()

    save_dir = "calibration_images"
    os.makedirs(save_dir, exist_ok=True)

    # Saving every 2 seconds
    last_save_time = time.time()
    save_interval = 2
    image_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Disconnected")
            break
        
        current_time = time.time()
        
        # Save every 2 seconds
        if current_time - last_save_time >= save_interval:
            filename = os.path.join(save_dir, f"image_{image_count:04d}.png")
            cv2.imwrite(filename, frame)
            print(f"Saved: {filename}")
            image_count += 1
            last_save_time = current_time
        
        # Put information in frame
        time_remaining = save_interval - (current_time - last_save_time)
        cv2.putText(frame, f"Next capture in: {time_remaining:.1f}s", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Images saved: {image_count}", 
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Camera Feed', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nTotal images captured: {image_count}")

if __name__ == "__main__":
    action = input("1: extract calibration data\n2: extract calibration coefficients\n3: view calibration data\n> ")

    if action == "1":
        capture_images()

    elif action == "2":
        # Use the same directory we saved to
        img_dir = "calibration_images"
        imgs = sorted(os.listdir(img_dir))

        # Getting the corners from all the images
        print("\nProcessing images for calibration...")
        corners_list, pattern_points_list, h, w = get_img_info(imgs, img_dir)

        if corners_list is None or len(corners_list) < 3:
            print("Not enough valid images for calibration!")
            exit()

        # Calibrate the camera based on all the images
        print(f"\nCalibrating camera with {len(corners_list)} images...")
        ret, K, d, rvecs, tvecs, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors = cv2.calibrateCameraExtended(
            pattern_points_list, corners_list, (w, h), None, None
        )

        calibration_data = {
            'camera_matrix': K,
            'distortion_coeffs': d,
            'rotation_vectors': rvecs,
            'translation_vectors': tvecs,
            'std_intrinsics': stdDeviationsIntrinsics,
            'std_extrinsics': stdDeviationsExtrinsics,
            'per_view_errors': perViewErrors,
            'reprojection_error': ret,
            'image_size': (w, h)
        }
        
        with open("calibration_coefficients.pkl", "wb") as f:
            pickle.dump(calibration_data, f)
        
        print("Finished calibration\n")
        print(f"Reprojection error: {ret:.4f}")
        print(f"\nCamera matrix (K):\n{K}")
        print(f"\nDistortion coefficients (d):\n{d}")

    elif action == "3":
        if not os.path.exists("calibration_coefficients.pkl"):
            print("calibration_coefficients.pkl not found")
            exit()

        with open("calibration_coefficients.pkl", "rb") as f:
            data = pickle.load(f)

        K = data['camera_matrix']
        d = data['distortion_coeffs']
        ret = data['reprojection_error']
        img_size = data['image_size']

        print("Finished calibration\n")
        print(f"Reprojection error: {ret:.4f}")
        print(f"Image size: {img_size}")
        print(f"\nCamera matrix (K):\n{K}")
        print(f"\nDistortion coefficients (d):\n{d}")
    
    else:
        pass