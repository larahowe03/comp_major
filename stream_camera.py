import cv2
import pickle
from warp_board import process_chess_image
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np

def contour_detect(im):
    # Convert to grayscale
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to reduce noise while keeping edges sharp
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Apply CLAHE for better contrast
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # enhanced = clahe.apply(filtered)
    
    # Adaptive thresholding for better edge detection
    binary = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    
    # Morphological operations to clean up
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
    
    # Edge detection with optimized parameters
    edges = cv2.Canny(morph, 50, 150)
    
    # Find contours
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Intelligent filtering
    filtered_contours = []
    for c in contours:
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        
        # Filter by area and aspect ratio
        if area > 100:
            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = float(w) / h if h > 0 else 0
            
            # Keep contours with reasonable aspect ratios (not too thin)
            if 0.2 < aspect_ratio < 5.0:
                filtered_contours.append(c)
    
    # Draw contours with different colors based on size
    result = im.copy()
    for c in filtered_contours:
        area = cv2.contourArea(c)
        # Color code by size: small=green, medium=blue, large=red
        if area < 500:
            color = (0, 255, 0)  # Green
        elif area < 2000:
            color = (255, 0, 0)  # Blue
        else:
            color = (0, 0, 255)  # Red
        
        cv2.drawContours(result, [c], -1, color, 2)
    
    return result

def mask_brown(im):
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    
    # Define range for light brown/tan colors
    # Hue: 10-30 (orange-brown range)
    # Saturation: 30-150 (not too grey, not too saturated)
    # Value: 100-200 (light to medium brightness)
    lower_brown = np.array([10, 30, 100])
    upper_brown = np.array([30, 150, 200])
    
    # Create mask for brown pixels
    mask = cv2.inRange(hsv, lower_brown, upper_brown)
    
    # Invert mask to keep everything EXCEPT brown
    mask_inv = cv2.bitwise_not(mask)
    
    # Apply mask to original image
    result = cv2.bitwise_and(im, im, mask=mask_inv)
    
    return result
def colour_space(im):
    # Convert to HSV
    im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(im_hsv)
    
    # Prepare pixel colors for visualization
    height, width = im.shape[:2]
    pixel_colours = im.reshape((height * width, 3)) / 255.0  # Normalize to 0-1
    pixel_colours = pixel_colours[:, ::-1]  # BGR to RGB
    
    # Enable interactive mode
    plt.ion()
    
    # Create or clear figure
    plt.clf()
    fig = plt.gcf()
    fig.set_size_inches(12, 10)
    axis = fig.add_subplot(1, 1, 1, projection="3d")
    
    # Downsample for performance
    step = max(1, (height * width) // 10000)
    
    axis.scatter(H.flatten()[::step], 
                 S.flatten()[::step], 
                 V.flatten()[::step], 
                 c=pixel_colours[::step], 
                 marker='.', 
                 s=1,
                 alpha=0.5)
    
    axis.set_xlabel("Hue", fontsize=12)
    axis.set_ylabel("Saturation", fontsize=12)
    axis.set_zlabel("Value", fontsize=12)
    axis.set_title("HSV Color Space", fontsize=14)
    axis.view_init(elev=30, azim=45)
    
    plt.tight_layout()
    plt.draw()
    plt.pause(0.001)  # Small pause to update display

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

        # colour_space(clahed)

        browk_mask = mask_brown(processed)

        img = contour_detect(clahed)

        if ret:
            cv2.imshow('Processed', processed)
            # cv2.imwrite("blah.png", processed)
            # yolod = resynet_test(processed)
        

        cv2.imshow('Undistorted', frame_undistorted)
        cv2.imshow('Distorted', frame)
        cv2.imshow('clahed', clahed)
        cv2.imshow('browk_mask', browk_mask)
        cv2.imshow('img', img)
        # cv2.imshow('hsv', hsv)
        # cv2.imshow('thresh', thresh)
        # if yolod is not None and yolod.size > 0:
        #     cv2.imshow('yolod', yolod)
        # else:
        #     print("⚠️ Skipping frame — yolod is empty or invalid.")

        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

