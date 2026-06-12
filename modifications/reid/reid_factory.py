from .osnet_reid import OSNetReID
from .resnet50_reid import ResNet50ReID
from .mobilenet_reid import MobileNetReID

def build_reid(name):
    if name == "osnet":
        return OSNetReID("osnet_x1_0")
    elif name == "resnet50":
        return ResNet50ReID("resnet50")
    elif name == "mobilenet":
        return MobileNetReID("mobilenetv2_x1_0")
    
    raise ValueError(f"Unknown REID model: {name}")