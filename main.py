from torchvision import models
import xml.etree.ElementTree as et
import os
from config import *




label_path = '/Users/andriizelenko/qvuer7/projects/CV_main_tasks/data_detection/test_zip/test/apple_77.xml'
tree = et.parse(label_path)

root = tree.getroot()
labels = []
boxes = []
for member in root.findall('object'):
    labels.append(member.find('name').text)
    xmin = int(member.find('bndbox').find('xmin').text)
    # xmax = right corner x-coordinates
    xmax = int(member.find('bndbox').find('xmax').text)
    # ymin = left corner y-coordinates
    ymin = int(member.find('bndbox').find('ymin').text)
    # ymax = right corner y-coordinates
    ymax = int(member.find('bndbox').find('ymax').text)
    boxes.append([xmin, ymin, xmax, ymax])


print(CHECKPOINTS_PATH)



model = models.detection.fasterrcnn_resnet50_fpn(pretrained = True, n_classes  = 2)
epoch = 12
torch.save(model.state_dict(), CHECKPOINTS_PATH + 'checkpoint_' + str(epoch) + '.pt')


'''

#pytorch_total_params = sum(p.numel() for p in model.parameters())




# freeze the backbone (it will freeze the body and fpn params)
for p in model.backbone.parameters():
  p.requires_grad = False

# freeze the fc6 layer in roi_heads
for p in model.roi_heads.box_head.fc6.parameters():
  p.requires_grad = False
'''


