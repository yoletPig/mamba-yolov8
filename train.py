from ultralytics import YOLO

model = YOLO('ultralytics/cfg/models/v8/yolov8s-macaque-pose.yaml',task="pose")
model.train(data='ultralytics/cfg/datasets/macaque-pose.yaml', epochs=100, imgsz=640,batch=32,workers=1,val=False,resume=True)