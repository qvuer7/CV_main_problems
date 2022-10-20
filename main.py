from utils.trainers import *
from utils.data_loading import *
from utils.models import *
import torch
from utils.vizualisation import *
import cv2
from torch import tensor
from config import *
import matplotlib.pyplot as plt






checkpoint_path = '/Users/andriizelenko/qvuer7/projects/CV_main_tasks/checkpoints/detection/checkpoint_detector.pth'

train_df, test_df = get_dataframes()
train_transform, test_transform = get_detection_transforms()
md = DetectionDataset(train_df, transform = test_transform ,labels_map = LABELS_MAP)
image_ori, target = md[200]


output = inference_faster_rcnn(checkpoint_path = checkpoint_path, image_ori = image_ori)
image = draw_bounding_box_from_ITtensor(image_s=image_ori, target=output, label_map=LABELS_MAP)
image_2 = draw_bounding_box_from_ITtensor(image_s=image_ori, target=target, label_map=LABELS_MAP)
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(image_2)
ax1.set_title('original image')
ax2.imshow(image)
ax2.set_title('output from detector')
plt.show()




