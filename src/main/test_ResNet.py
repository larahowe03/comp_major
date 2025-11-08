import torch
from torchvision import models, transforms
import torch.nn.functional as F
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from warp_board import process_chess_image   # your warp function

# ----------------------------
# 1️⃣ Configuration
# ----------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
class_names = [
    'black_bishop', 'black_king', 'black_knight', 'black_pawn',
    'black_queen', 'black_rook',
    'white_bishop', 'white_king', 'white_knight', 'white_pawn',
    'white_queen', 'white_rook'
]
num_classes = len(class_names)
model_path = 'models/ResNet/final_resnet_chess_best.pth'
confidence_threshold = 0.7    # adjust based on your model’s reliability
brightness_threshold = 220    # bright = probably empty
contrast_threshold = 15       # low texture = probably empty
margin = 80                   # pixels on each side

# ----------------------------
# 2️⃣ Load model
# ----------------------------
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# ----------------------------
# 3️⃣ Define transform (same as val)
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ----------------------------
# 4️⃣ Load and warp the image
# ----------------------------
img = cv2.imread("image.png")
warp_margined, warp_unmargined, contoured_img, pts_src = process_chess_image(img)

if warp_margined is None:
    print("⚠️ Board could not be warped.")
    exit()

# Optional visual check
plt.figure(figsize=(12,4))
plt.subplot(1,3,1); plt.imshow(cv2.cvtColor(contoured_img, cv2.COLOR_BGR2RGB)); plt.title("Detected Corners")
plt.subplot(1,3,2); plt.imshow(warp_unmargined); plt.title("Unmargined Board")
plt.subplot(1,3,3); plt.imshow(warp_margined); plt.title("Margined Board")
plt.show()

# ----------------------------
# 5️⃣ Crop 8x8 grid & classify (ignore empty)
# ----------------------------
h, w = warp_margined.shape[:2]
rows, cols = 8, 8

usable_h = h - 2 * margin
usable_w = w - 2 * margin
cell_h = usable_h // rows
cell_w = usable_w // cols

predicted_board = []
overlay = warp_margined.copy()

for i in range(rows):
    row_preds = []
    for j in range(cols):
        y1 = margin + i * cell_h
        y2 = margin + (i + 1) * cell_h
        x1 = margin + j * cell_w
        x2 = margin + (j + 1) * cell_w

        cell = warp_margined[y1:y2, x1:x2]

        # Quick brightness/contrast heuristic for empty squares
        gray = cv2.cvtColor(cell, cv2.COLOR_RGB2GRAY)
        if gray.mean() > brightness_threshold or gray.std() < contrast_threshold:
            row_preds.append(None)
            continue

        # Model prediction
        cell_pil = Image.fromarray(cell)
        tensor = transform(cell_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(tensor)
            probs = F.softmax(output, dim=1)
            conf, pred = probs.max(1)
            conf_val = conf.item()
            pred_label = class_names[pred.item()]

        # Skip low-confidence predictions
        if conf_val < confidence_threshold:
            row_preds.append(None)
            continue

        # Save prediction
        row_preds.append(pred_label)
        cv2.putText(
            overlay, f"{pred_label} ({conf_val:.2f})",
            (x1 + 5, y1 + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 1
        )
    predicted_board.append(row_preds)

# ----------------------------
# 6️⃣ Show results
# ----------------------------
plt.figure(figsize=(8,8))
plt.imshow(overlay)
plt.title("Predicted Chessboard (Non-Empty Only)")
plt.axis('off')
plt.show()

print("\nPredicted non-empty squares (top to bottom):")
for row in predicted_board:
    print(row)
