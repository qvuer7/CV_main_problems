from utils.data_loading import *
from utils.models import *
from utils.trainers import *
from config import *


train_loader, test_loader = get_detection_data_loaders()
model = get_faster_rcnn()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
train_detector(model = model, trainLoader = train_loader, testLoader = test_loader, optimizer = optimizer)




