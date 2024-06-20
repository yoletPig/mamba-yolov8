from ultralytics import YOLO
import torch

model = YOLO('ultralytics/cfg/models/v8/yolov8n-macaque-pose.yaml',task="pose")
print(model.info(detailed=True))