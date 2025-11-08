import pandas as pd
import matplotlib.pyplot as plt

resnet_losses = pd.read_csv("models/YOLOv8/final_model_warp_colour/results.csv")
best_epoch = 10
plt.plot(resnet_losses["epoch"], resnet_losses["metrics/mAP50(B)"])
resnet_losses = pd.read_csv("runs_chess/model_warp_contour/results.csv")
plt.plot(resnet_losses["epoch"], resnet_losses["metrics/mAP50(B)"])
plt.legend(["Colour accuracy", "Contour accuracy"])
plt.title("YOLOv8x validation accuracy over epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.savefig("training_yolo.png")