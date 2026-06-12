import torch
import cv2
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms import functional as F
from .base_detector import BaseDetector

class FasterRCNNDetector(BaseDetector):

    def __init__(self, weights="DEFAULT", conf_thresh=0.5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.conf_thresh = conf_thresh
        
        if weights == "DEFAULT":
            self.model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        else:
            self.model = fasterrcnn_resnet50_fpn(weights=weights)
            
        self.model.to(self.device)
        self.model.eval()

    def detect(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = F.to_tensor(frame_rgb).to(self.device)
        
        with torch.no_grad():
            predictions = self.model([img_tensor])[0]
        
        detections = []
        
        for i, label in enumerate(predictions['labels']):
            if label == 1:
                x1, y1, x2, y2 = predictions['boxes'][i].cpu().numpy()
                conf = float(predictions['scores'][i].cpu().numpy())

                if conf >= self.conf_thresh:
                    detections.append([x1, y1, x2, y2, conf])

        return detections