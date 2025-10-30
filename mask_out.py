import cv2
import numpy as np
import matplotlib.pyplot as plt
from warp_board import process_chess_image

def mask_out(img):
    # -------------------------
    # Load warped chessboard image
    # -------------------------
    # img = cv2.imread("img.png")
    # warp = process_chess_image(img)
    # warp2 = process_chess_image(warp)
    # warp2 = cv2.resize(warp2, (800, 800))

    # -------------------------
    # Otsu threshold (for black/white separation)
    # -------------------------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # -------------------------
    # Iterate over 8x8 squares
    # -------------------------
    h, w = gray.shape
    rows, cols = 8, 8
    cell_h, cell_w = h // rows, w // cols

    result = img.copy()

    for i in range(rows):
        for j in range(cols):
            y1, y2 = i*cell_h, (i+1)*cell_h
            x1, x2 = j*cell_w, (j+1)*cell_w

            # Crop the thresholded square
            cell_thresh = otsu[y1:y2, x1:x2]

            # Count black vs white pixels
            black_count = np.sum(cell_thresh == 0)
            white_count = np.sum(cell_thresh == 255)

            # Decide majority (0 = black, 255 = white)
            if black_count > white_count:
                majority_val = 0
            else:
                majority_val = 255

            # Build a mask of majority pixels
            cell_mask = (cell_thresh == majority_val).astype(np.uint8) * 255

            # Paint majority pixels red on result
            result[y1:y2, x1:x2][cell_mask == 255] = (0, 0, 255)

    # -------------------------
    # Show results
    # -------------------------
    # plt.figure(figsize=(15,6))
    # plt.subplot(1,3,1); plt.title("Original"); plt.imshow(cv2.cvtColor(warp2, cv2.COLOR_BGR2RGB)); plt.axis("off")
    # plt.subplot(1,3,2); plt.title("Otsu Threshold"); plt.imshow(otsu, cmap="gray"); plt.axis("off")
    # plt.subplot(1,3,3); plt.title("Red Masked Board (pieces untouched)"); plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)); plt.axis("off")
    # plt.show()

    # cv2.imwrite("chess_board_majority_red.png", result)
    return result
