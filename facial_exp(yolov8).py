from roboflow import Roboflow
import ultralytics
from ultralytics import YOLO
import os
from IPython.display import display, Image
display.clear_output()
ultralytics.checks()
import torch

# Check if CUDA (GPU) is available
if torch.cuda.is_available():
    # Enable memory growth for all available GPUs
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_per_process_memory_growth(i, True)

        
model = YOLO("yolov8s.pt")

# yolo task=detect mode=train model=yolov8s.pt data=data.yaml epochs=100 imgsz=640


