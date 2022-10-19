from utils.trainers import *
from utils.data_loading import *
from utils.models import *
import torch
from utils.vizualisation import *
import cv2

checkpoint_path = '/Users/andriizelenko/qvuer7/projects/CV_main_tasks/checkpoints/detection/checkpoint_detector.pth'

train_df, test_df = get_dataframes()
print(train_df)
train_transform, test_transform = get_detection_transforms()
md = DetectionDataset(test_df, transform = test_transform ,labels_map = LABELS_MAP)
image_ori, target = md[5]

image = image_ori.clone()
image = image.unsqueeze(0)

model, _ = get_faster_rcnn()
model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))

model.eval()

image = image.to(DEVICE)
model = model.to(DEVICE)
with torch.no_grad():
    output = model(image)


output = output[0]

image = draw_bounding_box_from_ITtensor(image = image_ori, target = output, label_map=LABELS_MAP)
print(output)
cv2.imshow('image', image)
cv2.waitKey(0)



''' EXAMPLE
 train_df, test_df = get_dataframes()
 _, test_transforms = get_segmentation_transforms()
 dataset = segmentationDataset(dataFrame = test_df, transform = test_transforms)
 image, mask = dataset[1]
 output_mask = inference_deep_lab(image_m = image, checkpoint_path = 'custom/checkpoint/path, threshold = custom.threshold)
 vizualize_segmentation_output(image_ori = image, mask_ori = mask, mask = output_mask)
 '''
