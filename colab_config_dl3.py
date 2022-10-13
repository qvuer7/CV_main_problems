import torch
from torch.utils.tensorboard import SummaryWriter


DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
NUM_EPOCHS = 20
BATCH_SIZE = 9
LR = 0.01
IMAGE_HEIGH = 512
IMAGE_WIDTH = 512
IMAGES_PATH = '/content/drive/MyDrive/Photos data/data_segmentation/images/'
ANNOTATIONS_PATH = '/content/drive/MyDrive/Photos data/data_segmentation/annotations/'
TRAIN_RATIO = 0.85
LOG_DIR = '/content/CV_main_problems/logs/'
N_CLASSES = 2
CHECKPOINTS_PAHT = '/content/CV_main_problems/checkpoints/'
writer = SummaryWriter()

