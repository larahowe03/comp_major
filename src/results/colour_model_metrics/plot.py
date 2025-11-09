import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ========================
# Load CSVs
# ========================
train = pd.read_csv("train_per_class_metrics.csv")
val = pd.read_csv("val_per_class_metrics.csv")
test = pd.read_csv("test_per_class_metrics.csv")

# Add labels
train["set"] = "Train"
val["set"] = "Validation"
test["set"] = "Test"

# Combine
df = pd.concat([train, val, test], ignore_index=True)

# ========================
# Helper Function
# ========================
def get_grouped_data(metric):
    """Return grouped values for train/val/test for the given metric."""
    train_vals = df[df["set"] == "Train"][metric].values
    val_vals = df[df["set"] == "Validation"][metric].values
    test_vals = df[df["set"] == "Test"][metric].values
    return train_vals, val_vals, test_vals

# ========================
# Plot Setup
# ========================
metrics = [
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("mAP50", "mAP@0.5"),
    ("mAP50-95", "mAP@0.5:0.95")
]

classes = df["class_name"].unique()
x = np.arange(len(classes))
width = 0.25

fig, axs = plt.subplots(2, 2, figsize=(12, 8))
axs = axs.flatten()

for ax, (metric, title) in zip(axs, metrics):
    train_vals, val_vals, test_vals = get_grouped_data(metric)
    
    ax.bar(x - width, train_vals, width, label="Train")
    ax.bar(x, val_vals, width, label="Validation")
    ax.bar(x + width, test_vals, width, label="Test")
    
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

# Shared title and labels
fig.suptitle("Contour Model – Per-Class Performance Metrics", fontsize=14, fontweight='bold', y=0.98)
fig.text(0.5, 0.04, "Class", ha="center", fontsize=12)
# fig.text(0.04, 0.5, "Score", va="center", rotation="vertical", fontsize=12)

# ========================
# Legend at bottom
# ========================
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc="lower center",
    ncol=3,
    bbox_to_anchor=(0.5, -0.02),
    frameon=False
)

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.show()
