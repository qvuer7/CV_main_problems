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


train_df, test_df = get_instance_segmentation_dataframes()

train_t, test_t = get_detection_transforms()
md = InstanceSegmentationDataset(train_df, train_t)

for i in range(len(md)):
    image, data = md[i]
    pass