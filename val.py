from ultralytics import YOLO

# model = YOLO('runs/pose/train31/weights/best.pt',task="pose")
model = YOLO('runs/pose/train32/weights/best.pt',task="pose")
# model = YOLO('ultralytics/cfg/models/v8/yolov8n-macaque-pose.yaml',task="pose")
model.val(data='ultralytics/cfg/datasets/macaque-pose.yaml',imgsz=640,batch=32,workers=1)