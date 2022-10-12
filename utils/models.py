from torchvision import models
from config import *

def get_deep_lab_v3():
    model = models.segmentation.deeplabv3_resnet50(pretrained=True, progress=True)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.classifier = models.segmentation.deeplabv3.DeepLabHead(2048, N_CLASSES)
    return model
