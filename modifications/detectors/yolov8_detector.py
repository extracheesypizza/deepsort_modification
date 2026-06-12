import torch
from ultralytics import YOLO
from .base_detector import BaseDetector

class YOLOv8Detector(BaseDetector):

    def __init__(self, weights="yolov8n.pt", conf_thresh=0.5):
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        self.conf_thresh = conf_thresh
        
        self.model = YOLO(weights)
        
        if self.device == 0:
            self.model.warmup(imgsz=(1, 3, 640, 640))

    def detect(self, frame):
        results = self.model(frame, device=self.device, verbose=False, imgsz=640)[0]
        detections = []

        for box in results.boxes:
            cls = int(box.cls)
            if cls != 0:
                continue
            
            conf = float(box.conf)
            
            if conf < self.conf_thresh:
                continue
                
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            detections.append([x1, y1, x2, y2, conf])

        return detections