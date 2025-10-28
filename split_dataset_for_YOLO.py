import os
import shutil
import random
from pathlib import Path

# -------------------------------
# CONFIGURATION
# -------------------------------
source_dir = Path("chess_piece_images_for_ResNet")
output_dir = Path("dataset_yolo")

train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1!"

# -------------------------------
# MAKE CLEAN DIRECTORIES
# -------------------------------
def make_clean_dir(path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

for split in ["train", "val", "test"]:
    make_clean_dir(output_dir / "images" / split)
    make_clean_dir(output_dir / "labels" / split)

# -------------------------------
# CLASS NAME MAPPING
# -------------------------------
class_names = sorted([d.name for d in source_dir.iterdir() if d.is_dir()])
class_to_id = {name: i for i, name in enumerate(class_names)}

print("🧩 Class mapping:")
for name, idx in class_to_id.items():
    print(f"{idx}: {name}")

# -------------------------------
# SPLIT + GENERATE LABELS
# -------------------------------
for class_dir in source_dir.iterdir():
    if not class_dir.is_dir():
        continue

    class_id = class_to_id[class_dir.name]
    images = [f for f in class_dir.glob("*.*") if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    random.shuffle(images)

    n_total = len(images)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)

    splits = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:]
    }

    for split_name, split_files in splits.items():
        for img_path in split_files:
            dest_img_dir = output_dir / "images" / split_name
            dest_lbl_dir = output_dir / "labels" / split_name
            dest_img_dir.mkdir(parents=True, exist_ok=True)
            dest_lbl_dir.mkdir(parents=True, exist_ok=True)

            # copy image
            shutil.copy(img_path, dest_img_dir / img_path.name)

            # create YOLO label file
            label_path = dest_lbl_dir / f"{img_path.stem}.txt"
            with open(label_path, "w") as f:
                # YOLO expects: class_id x_center y_center width height (all normalized)
                # For classification, we don’t have bounding boxes, so we can use full frame (0.5 0.5 1 1)
                f.write(f"{class_id} 0.5 0.5 1 1\n")

    print(f"✅ {class_dir.name}: {n_total} images split into train/val/test and labeled.")

print("\n🎯 YOLO dataset with auto-generated labels created at:")
print(f"📁 {output_dir}/images/")
print(f"📁 {output_dir}/labels/")

# -------------------------------
# CREATE data.yaml FILE
# -------------------------------
yaml_path = output_dir / "data.yaml"
with open(yaml_path, "w") as f:
    f.write(f"train: {output_dir}/images/train\n")
    f.write(f"val: {output_dir}/images/val\n")
    f.write(f"test: {output_dir}/images/test\n\n")
    f.write(f"nc: {len(class_names)}\n")
    f.write("names: [\n")
    for i, name in enumerate(class_names):
        comma = "," if i < len(class_names) - 1 else ""
        f.write(f"  '{name}'{comma}\n")
    f.write("]\n")

print(f"\n🧾 data.yaml generated at: {yaml_path}")
