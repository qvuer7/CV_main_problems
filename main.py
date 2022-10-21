import sklearn.model_selection

from utils.trainers import *
from utils.data_loading import *
from utils.models import *
import torch
from utils.vizualisation import *
import cv2
from torch import tensor
from config import *
import matplotlib.pyplot as plt
import pandas as pd
import os
from torch.utils.data import Dataset
import albumentations as A



checkpoint_path = '/Users/andriizelenko/qvuer7/projects/CV_main_tasks/checkpoints/instance_segmentation/checkpoint_instance_segmentation.pth'
model, _ = get_mask_rcnn()



model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
model.eval()

train_df, test_df = get_instance_segmentation_dataframes()
train_t, test_t = get_detection_transforms()
md = InstanceSegmentationDataset(test_df, transforms = test_t, labels_map = LABELS_MAP)
i = 2
image_m, target = md[i]
image = image_m.clone().detach()
image = image.unsqueeze(0)
image = image.to(DEVICE)


with torch.no_grad():
    out = model(image)

print(out)
out = out[0]
image = draw_bounding_box_from_ITtensor(image_s = image_m, target = out, label_map = LABELS_MAP)
fig, (ax1, ax2) = plt.subplots(1,2)
ax1.imshow(image)
ax2.imshow(image)
plt.show()