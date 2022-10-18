from torchvision import models
import xml.etree.ElementTree as et
import os
from config import *
import cv2
import albumentations as A
from utils.vizualisation import *
from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from utils.data_loading import *
from tqdm import tqdm





train_loader, test_loader = get_detection_data_loaders()



model = models.detection.fasterrcnn_resnet50_fpn(pretrained = True, n_classes  = 2)

#pytorch_total_params = sum(p.numel() for p in model.parameters()
# freeze the backbone (it will freeze the body and fpn params)
for p in model.backbone.parameters():
  p.requires_grad = False

# freeze the fc6 layer in roi_heads
for p in model.roi_heads.box_head.fc6.parameters():
  p.requires_grad = False


for i, data in enumerate(tqdm(train_loader)):
    images, targets = data
    images = list(image.to(DEVICE) for image in images)
    targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
    break


model.train()
loss = model(images, targets)
print(loss)
