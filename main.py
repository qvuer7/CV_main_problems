from torchvision import models
import xml.etree.ElementTree as ET



'''
model = models.detection.fasterrcnn_resnet50_fpn(pretrained = True, n_classes  = 2)


#pytorch_total_params = sum(p.numel() for p in model.parameters())




# freeze the backbone (it will freeze the body and fpn params)
for p in model.backbone.parameters():
  p.requires_grad = False

# freeze the fc6 layer in roi_heads
for p in model.roi_heads.box_head.fc6.parameters():
  p.requires_grad = False
'''


