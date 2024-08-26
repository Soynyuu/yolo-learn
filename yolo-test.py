from ultralytics import YOLO
import cv2

# モデル読み込み
model = YOLO("runs/detect/train/weights/last.pt")

# 入力画像
results = model('img-path',save=True)