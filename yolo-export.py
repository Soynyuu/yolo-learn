from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # load an official model
model = YOLO("/workspaces/yolo-learn/runs/detect/train/weights/best.pt")  # load a custom trained model

# Export the model
model.export(format="tfjs")