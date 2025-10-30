from warp_board import process_chess_image
import cv2
import matplotlib.pyplot as plt
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np

def resynet_test(img):
    # img = cv2.imread("img.png")
    # path = "img.png"

    # warp = process_chess_image(path)

    # plt.imshow(warp)
    # plt.show()

    # 1. Load the trained model
    classes = ['black_bishop', 'black_king', 'black_knight', 'black_pawn', 'black_queen', 'black_rook', 'empty',
            'white_bishop', 'white_king', 'white_knight', 'white_pawn', 'white_queen', 'white_rook']

    model = models.resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(torch.load('resnet_chess_finetuned.pkl', map_location='cpu'))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(15),             # small random rotations
        transforms.ColorJitter(brightness=0.2,
                            contrast=0.2,
                            saturation=0.2),   # lighting variation
        transforms.RandomHorizontalFlip(),         # mirror variations
        transforms.RandomVerticalFlip(),           # optional (if top-down view)
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    # Make a copy of the input image for annotations
    annotated_img = img.copy()
    
    h, w = img.shape[:2]
    rows, cols = 8, 8
    cell_h, cell_w = h // rows, w // cols

    predicted_board = []

    for i in range(rows):
        row_preds = []
        for j in range(cols):
            y1, y2 = i * cell_h, (i + 1) * cell_h
            x1, x2 = j * cell_w, (j + 1) * cell_w
            cell = img[y1:y2, x1:x2]

            # Optional: skip empty cells (simple threshold)
            if cell.mean() < 30:  # tweak based on lighting
                pred_class = "empty"
            else:
                # Convert cell to PIL Image
                cell_pil = Image.fromarray(cv2.cvtColor(cell, cv2.COLOR_BGR2RGB))
                img_t = transform(cell_pil).unsqueeze(0)

                with torch.no_grad():
                    output = model(img_t)
                    pred_idx = output.argmax(1).item()
                    pred_class = classes[pred_idx]

            row_preds.append(pred_class)

            # -------------------------
            # 5️⃣ Overlay text on board
            # -------------------------
            label = pred_class.replace("white_", "W_").replace("black_", "B_")
            color = (0, 0, 255) if "black" in pred_class else (255, 255, 255)
            if pred_class != "empty":
                cv2.putText(annotated_img, label, (x1 + 5, y2 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        
        predicted_board.append(row_preds)

    # -------------------------
    # 6️⃣ Display results
    # -------------------------
    # cv2.imshow("Predicted Chessboard", annotated_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return annotated_img