import numpy as np
import cv2
import glob

def calibrate_camera(images_folder="calibration_images/*.jpg", grid_size=(7, 7)):
    """
    Calibrate camera using multiple chessboard images
    
    Args:
        images_folder: Path to calibration images
        grid_size: Internal corners on your chessboard (e.g., 7x7 for 8x8 board)
    """
    
    # Termination criteria for corner refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # Prepare object points (0,0,0), (1,0,0), (2,0,0) ...
    objp = np.zeros((grid_size[0] * grid_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:grid_size[0], 0:grid_size[1]].T.reshape(-1, 2)
    
    # Arrays to store object points and image points
    objpoints = []  # 3D points in real world
    imgpoints = []  # 2D points in image plane
    
    # Load images
    images = glob.glob(images_folder)
    
    if len(images) == 0:
        print(f"✗ No images found in {images_folder}")
        return None
    
    print(f"Found {len(images)} images")
    
    successful = 0
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, grid_size, None)
        
        if ret:
            successful += 1
            print(f"✓ {fname}")
            
            objpoints.append(objp)
            
            # Refine corner locations
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            
            # Draw and save (optional)
            img_drawn = cv2.drawChessboardCorners(img, grid_size, corners2, ret)
            cv2.imwrite(f"detected_{fname.split('/')[-1]}", img_drawn)
        else:
            print(f"✗ {fname}")
    
    if successful < 10:
        print(f"\n⚠️ Warning: Only {successful} images successful. Recommended: 15-20 images")
    
    if successful == 0:
        print("✗ No successful detections. Cannot calibrate.")
        return None
    
    print(f"\n{'='*50}")
    print(f"Calibrating with {successful} images...")
    print(f"{'='*50}\n")
    
    # Calibrate camera
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )
    
    if ret:
        print("✓ Calibration successful!\n")
        
        # Extract parameters
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]
        
        print("Camera Matrix (Intrinsic Parameters):")
        print(camera_matrix)
        print(f"\nFocal Length:")
        print(f"  fx = {fx:.2f} pixels")
        print(f"  fy = {fy:.2f} pixels")
        print(f"\nPrincipal Point (optical center):")
        print(f"  cx = {cx:.2f} pixels")
        print(f"  cy = {cy:.2f} pixels")
        print(f"\nDistortion Coefficients:")
        print(f"  k1 = {dist_coeffs[0][0]:.6f}")
        print(f"  k2 = {dist_coeffs[0][1]:.6f}")
        print(f"  p1 = {dist_coeffs[0][2]:.6f}")
        print(f"  p2 = {dist_coeffs[0][3]:.6f}")
        print(f"  k3 = {dist_coeffs[0][4]:.6f}")
        
        # Calculate reprojection error
        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], 
                                             camera_matrix, dist_coeffs)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error
        
        mean_error /= len(objpoints)
        print(f"\nReprojection Error: {mean_error:.4f} pixels")
        print("(Lower is better. < 0.5 is excellent, < 1.0 is good)")
        
        # Save to file
        np.savez('camera_calibration.npz', 
                 camera_matrix=camera_matrix, 
                 dist_coeffs=dist_coeffs)
        print(f"\n✓ Saved calibration to 'camera_calibration.npz'")
        
        return camera_matrix, dist_coeffs
    
    return None

def load_calibration():
    """Load previously saved calibration"""
    try:
        data = np.load('camera_calibration.npz')
        return data['camera_matrix'], data['dist_coeffs']
    except:
        print("✗ No calibration file found. Run calibration first.")
        return None, None

def undistort_image(img, camera_matrix, dist_coeffs):
    """Remove lens distortion from image"""
    h, w = img.shape[:2]
    
    # Get optimal new camera matrix
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h)
    )
    
    # Undistort
    undistorted = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)
    
    # Crop to ROI
    x, y, w, h = roi
    undistorted = undistorted[y:y+h, x:x+w]
    
    return undistorted

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "calibrate":
        # Calibration mode
        print("="*50)
        print("CAMERA CALIBRATION")
        print("="*50)
        print("\nInstructions:")
        print("1. Take 15-20 photos of your chessboard from different angles")
        print("2. Include various distances and orientations")
        print("3. Keep the entire board visible in each photo")
        print("4. Save photos in 'calibration_images/' folder")
        print("5. Run: python script.py calibrate\n")
        
        camera_matrix, dist_coeffs = calibrate_camera()
    
    elif len(sys.argv) > 1 and sys.argv[1] == "test":
        # Test undistortion
        camera_matrix, dist_coeffs = load_calibration()
        
        if camera_matrix is not None:
            img = cv2.imread("test.png")
            if img is not None:
                undistorted = undistort_image(img, camera_matrix, dist_coeffs)
                cv2.imwrite("undistorted.png", undistorted)
                print("✓ Saved undistorted.png")
    
    else:
        print("Usage:")
        print("  python script.py calibrate  - Calibrate camera")
        print("  python script.py test       - Test undistortion on test.png")

    print(camera_matrix)
    print(dist_coeffs)