from torchvision import models
from utils.data_loading import segmentationDataset
from sklearn.model_selection import train_test_split
import os
import albumentations as A
import pandas as pd
from torch.utils.data import DataLoader
import torch
import numpy as np

data_path = 'data_segmentation/'
images_path = data_path + 'images/'
annotations_path = data_path + 'annotations/'
annotations = os.listdir(annotations_path)
images = os.listdir(images_path)
images_full = list(map(lambda x: images_path + x, images))
annotations_full = list(map(lambda x: annotations_path + x, annotations))
images_full.sort()
annotations_full.sort()
dataset_df = pd.DataFrame({'images': images_full, 'masks': annotations_full})
train_df, test_df = train_test_split(dataset_df, train_size=0.85, shuffle=False)
image_heigh = 512
image_width = 512



train_transform = A.Compose([
    A.LongestMaxSize(image_heigh),
    A.HorizontalFlip(p=0.5)

])

test_transform = A.Compose([
    A.Resize(image_heigh, image_width)

])





train_ds = segmentationDataset(train_df, train_transform)

a = DataLoader(train_ds, batch_size=1)
for i, (image, mask) in enumerate(a):
    break


model = models.segmentation.deeplabv3_resnet50(pretrained = True, num_classes = 2)

'''
model.train()
optimizer = torch.optim.Adam(model.parameters(), 0.0001)

'''
print(model)





