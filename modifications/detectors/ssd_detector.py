import torch
import cv2
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
from torchvision.transforms import functional as F
from .base_detector import BaseDetector

class SSDDetector(BaseDetector):

    def __init__(self, weights="DEFAULT", conf_thresh=0.5):
    
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[SSD] Initializing model on device: {self.device}")
        
        self.conf_thresh = conf_thresh
        
        if weights == "DEFAULT":
            self.model = ssdlite320_mobilenet_v3_large(weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT)
        else:
            self.model = ssdlite320_mobilenet_v3_large(weights=weights)
            
        self.model.to(self.device)
        self.model.eval()
        
        self.frame_counter = 0

    def detect(self, frame):
        self.frame_counter += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = F.to_tensor(frame_rgb).to(self.device)
        
        with torch.no_grad():
            predictions = self.model([img_tensor])[0]
        
        detections = []
        raw_person_count = 0
        
        for i, label in enumerate(predictions['labels']):
            if label == 1:  # 1 is 'person' in Torchvision COCO mapping
                raw_person_count += 1
                x1, y1, x2, y2 = predictions['boxes'][i].cpu().numpy()
                conf = float(predictions['scores'][i].cpu().numpy())
                
                if conf >= self.conf_thresh:
                    detections.append([x1, y1, x2, y2, conf])
        
        print(f"[SSD DEBUG @ Frame {self.frame_counter}] Raw person detections: {raw_person_count} | Passed threshold ({self.conf_thresh}): {len(detections)}")
        
        return detections