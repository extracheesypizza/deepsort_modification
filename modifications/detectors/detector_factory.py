from .yolov8_detector import YOLOv8Detector
from .fasterrcnn_detector import FasterRCNNDetector
from .ssd_detector import SSDDetector

def build_detector(name, conf_thresh):
    if name == "yolov8":
        return YOLOv8Detector("modifications/weights/yolov8n.pt", conf_thresh)
    elif name == "fasterrcnn":
        return FasterRCNNDetector("DEFAULT", conf_thresh)
    elif name == "ssd":
        return SSDDetector("DEFAULT", conf_thresh)
    
    raise ValueError(f"Unknown detector {name}")