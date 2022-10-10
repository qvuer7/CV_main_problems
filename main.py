from torchvision import models
from utils.data_loading import segmentationDataset
from sklearn.model_selection import train_test_split
import os
import albumentations as A
import pandas as pd
from torch.utils.data import DataLoader
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from tqdm import tqdm
import torch
import time

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
test_ds = segmentationDataset(test_df, test_transform)
train_loader = DataLoader(train_ds, batch_size=2)
test_loader = DataLoader(test_ds, batch_size = 2)


model = models.segmentation.deeplabv3_resnet50(pretrained = True, progress = True)

for parameter in model.parameters():
    parameter.requires_grad = False

model.classifier = DeepLabHead(2048, 1)


for i,(image, mask) in enumerate(train_loader):
    break





criterion = torch.nn.MSELoss(reduction='mean')
# Specify the optimizer with a lower learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
model.train()



device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


since = time.time()

image = image.to(device)
mask = mask.to(device)
optimizer.zero_grad()
out = model(image)
loss = criterion(out['out'], mask)
train_loss = loss.item()
loss.backward()
optimizer.step()

till = time.time()



def train_one_epoch(model, criterion, optimizer, dataLoader, device):
    model.train()
    total_loss = 0
    for i, (image, mask) in enumerate(tqdm(dataLoader)):
        image = image.to(device)
        mask = mask.to(device)
        optimizer.zero_grad()
        out = model(image)
        loss = criterion(out['out'], mask)
        total_loss+=loss.item()
        loss.backward()
        optimizer.step()
    return total_loss

def val_one_epoch(model,criterion, dataLoader, device):
    model.eval()
    total_loss = 0
    for i, (image, mask) in enumerate(tqdm(dataLoader)):
        image = image.to(device)
        mask = mask.to(device)
        out = model(image)
        loss = criterion(out['out'], mask)
        total_loss+=loss.item()

    return total_loss


train_loss = train_one_epoch(model = model, criterion = criterion, optimizer=optimizer,
                             device = device, dataLoader=train_loader)

print(train_loss)









