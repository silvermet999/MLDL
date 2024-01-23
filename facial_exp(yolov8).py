import ultralytics
from ultralytics import YOLO
import os
from IPython.display import clear_output, Image

from torch import cuda
from torch.cuda import empty_cache
import torchvision

clear_output()
empty_cache()
cuda.synchronize()
ultralytics.checks()
import torch


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

device = "cuda" if torch.cuda.is_available() else 'cpu'
print({device})
# torch.load("yolov8s.pt")
model = YOLO("yolov8s.yaml").to(device)

# !TORCH_CUDA_ARCH_LIST="3.5+PTX" TORCH_NVCC_FLAGS="-Xfatbin -compress-all" TORCH_USE_CUDA_DSA=1 yolo task=detect mode=train model=yolov8s.yaml data=data.yaml epochs=50 imgsz=60


