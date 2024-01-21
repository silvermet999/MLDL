from roboflow import Roboflow
import ultralytics
from ultralytics import YOLO
import os
from IPython.display import clear_output, Image
from torch.cuda import empty_cache
# ipc_collect -> Force collects GPU memory after it has been released by CUDA IPC.
# set_sync_debug_mode -> Sets the debug mode for cuda synchronizing operations.
# communication collectives

from torch import cuda
clear_output()
empty_cache()
cuda.synchronize()
ultralytics.checks()
import torch

# if torch.cuda.is_available():
#     for i in range(torch.cuda.device_count()):
#         set_per_process_memory_growth(i, True)

        
model = YOLO("yolov8s.pt")

!yolo task=detect mode=train model=yolov8s.pt data=data.yaml epochs=100 imgsz=640


