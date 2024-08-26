from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("/workspaces/yolo-learn/runs/detect/train/weights/best.pt")

# Export the model to TF.js format
model.export(format="onnx")  # creates '/yolov8n_web_model'
