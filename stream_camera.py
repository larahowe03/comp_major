import cv2
import pickle
from warp_board import process_chess_image
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors


def colour_space(im):
    height = im.shape[0]
    width = im.shape[1]
    pixel_colours = im.reshape((height*width, 3))
    norm = colors.Normalize(vmin=-1.0,vmax=1.0)
    norm.autoscale(pixel_colours)
    pixel_colours = norm(pixel_colours).tolist()
    # Convert to HSV
    im_hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)

    # visualise the colours in a RGB colour space
    H, S, V = cv2.split(im_hsv)
    
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1, projection="3d")
    axis.scatter(H.flatten(), S.flatten(), V.flatten(), facecolors=pixel_colours, marker='.')
    axis.set_xlabel("Hue")
    axis.set_ylabel("Saturation")
    axis.set_zlabel("Value")
    # axis.view_init(30,70,0) # (elevation, azimuth, roll): try adjusting to view from different perspectives

    plt.tight_layout()




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

