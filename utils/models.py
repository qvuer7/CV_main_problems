from torchvision import models
from config import *

def get_deep_lab_v3():
    model = models.segmentation.deeplabv3_resnet50(pretrained=True, progress=True)
    params  = []
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.classifier = models.segmentation.deeplabv3.DeepLabHead(2048, N_CLASSES)
    for p in model.parameters():
        if p.requires_grad:
            params.append(p)
    return model, params


def get_faster_rcnn():
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained = True, n_classes = N_CLASSES)
    params =[]
    for p in model.backbone.parameters():
        p.requires_grad = False

    # freeze the fc6 layer in roi_heads
    for p in model.roi_heads.box_head.fc6.parameters():
        p.requires_grad = False

    for p in model.parameters():
        if p.requires_grad:
            params.append(p)
    return model, params


def get_mask_rcnn():

    model = models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, N_CLASSES)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256

    model.roi_heads.mask_predictor = models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       N_CLASSES)

    for param in model.backbone.parameters():
        param.requires_grad = False

    params = [p for p in model.parameters() if p.requires_grad]

    return model, params

