from ultralytics import solutions

solutions.inference(model="runs/detect/train/weights/best.pt")