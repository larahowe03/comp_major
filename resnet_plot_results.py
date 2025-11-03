import pandas as pd
import matplotlib.pyplot as plt

resnet_losses = pd.read_csv("resnet_losses.csv")
best_epoch = 10
plt.plot(resnet_losses["epoch"], resnet_losses["train_acc"])
plt.plot(resnet_losses["epoch"], resnet_losses["val_acc"])
plt.axvline(x=best_epoch, color='red', linestyle='--', label=f'Best Epoch ({best_epoch})')
plt.legend(["Training accuracy", "Validation accuracy"])
plt.title("ResNet training accuracy over epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.savefig("training_resnet.png")

plt.clf()

resnet_losses = pd.read_csv("resnet_losses_finetuned.csv")
best_epoch = 8
plt.plot(resnet_losses["epoch"], resnet_losses["train_acc"])
plt.plot(resnet_losses["epoch"], resnet_losses["val_acc"])
plt.axvline(x=best_epoch, color='red', linestyle='--', label=f'Best Epoch ({best_epoch})')
plt.legend(["Training accuracy", "Validation accuracy"])
plt.title("ResNet finetuning accuracy over epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.savefig("finetuning_resnet.png")
