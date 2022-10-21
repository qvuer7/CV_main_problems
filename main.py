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





train_dl, test_dl = get_instance_segmentation_data_loaders()


model, params = get_mask_rcnn()
optimizer = torch.optim.SGD(params, lr=LR,
                                momentum=0.9, weight_decay=0.0005)
train_instance(model = model, trainLoader = train_dl, testLoader = test_dl, optimizer = optimizer)

