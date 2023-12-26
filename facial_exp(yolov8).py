from roboflow import Roboflow
import ultralytics
from ultralytics import YOLO
from IPython import display
import os
from IPython.display import display, Image
display.clear_output()
ultralytics.checks()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        
model = YOLO("yolov8s.pt")

# yolo task=detect mode=train model=yolov8s.pt data=data.yaml epochs=100 imgsz=640


