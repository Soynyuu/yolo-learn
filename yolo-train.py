from ultralytics import YOLO

# ベースとするモデル
model = YOLO('yolov8n.pt')

results = model.train(
    data='data.yaml-path', 
    epochs=3, 
    imgsz=480, 
    device='cpu'
)