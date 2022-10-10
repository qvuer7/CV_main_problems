import torch

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
NUM_EPOCHS = 20
BATCH_SIZE = 2
LR = 0.01
IMAGE_HEIGH = 512
IMAGE_WIDTH = 512
IMAGES_PATH = '/Users/andriizelenko/qvuer7/projects/CV_main_tasks/data_segmentation/images/'
ANNOTATIONS_PATH = '/Users/andriizelenko/qvuer7/projects/CV_main_tasks/data_segmentation/annotations/'
TRAIN_RATIO = 0.85