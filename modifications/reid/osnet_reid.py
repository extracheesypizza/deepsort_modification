import torch
import torch.nn.functional as F
import numpy as np
import torchreid
from .base_reid import BaseReID

class OSNetReID(BaseReID):

    def __init__(self, model_name="osnet_x1_0"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = torchreid.models.build_model(
            name=model_name,
            num_classes=0,
            loss='softmax',
            pretrained=True
        )
        self.model.to(self.device)
        self.model.eval()
        
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

    def extract_features(self, crops):
        if not crops:
            return np.array([])
        

        tensors = []
        for crop in crops:
            # crop is (H, W, 3) RGB uint8. Convert to (1, 3, H, W) float32 [0, 1] on GPU
            t = torch.from_numpy(crop).permute(2, 0, 1).float().to(self.device) / 255.0
            t = F.interpolate(t.unsqueeze(0), size=(256, 128), mode='bilinear', align_corners=False)
            tensors.append(t)
        
        batch = torch.cat(tensors, dim=0)
        
        batch = (batch - self.mean) / self.std
        
        with torch.no_grad():
            features = self.model(batch)
        
        features = torch.nn.functional.normalize(features, p=2, dim=1)
        return features.cpu().float().numpy()