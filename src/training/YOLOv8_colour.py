from ultralytics import YOLO
import torch
import shutil
from pathlib import Path

model_name = "yolov8n.pt" 
data_yaml = "datasets/dataset_yolo_warp_colour/data.yaml"
epochs = 500
img_size = 426
batch_size = 16
device = 0 if torch.cuda.is_available() else 'cpu'

print(f"Training YOLOv8 using {data_yaml}") 

# Load model
model = YOLO(model_name)

# Training parameters
results = model.train(
    data=data_yaml,      # path to data.yaml
    epochs=epochs,
    imgsz=img_size,
    batch=batch_size,
    device=device,
    name="final_model_warp_colour",
    project="models/YOLOv8",
    workers=2,
    optimizer='Adam',   
    lr0=0.0001,           
    patience=10,          
    verbose=True
)