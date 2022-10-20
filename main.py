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


train_df, test_df = get_instance_segmentation_dataframes()
i = 23
image = test_df.iloc[i].image
label = test_df.iloc[i].label
annotation = test_df.iloc[i].annotation

image = cv2.imread(image)
box = parse_pedd_fudan_annotation(annotation)

mask = cv2.imread(label)
print(np.unique(mask))



'''
for boxes in box:
    image = cv2.rectangle(image, (boxes[0], boxes[1]), (boxes[2], boxes[3]), (255, 0, 0), 2)

cv2.imshow('image', image)
cv2.waitKey(0)




fig, (ax1, ax2) = plt.subplots(1,2 )
image = cv2.imread(image)
label = cv2.imread(label)
label*=255
ax1.imshow(image)
ax2.imshow(label)
'''










