from utils.data_loading import get_data_loaders
from utils.trainers import train_one_epoch
from utils.models import get_deep_lab_v3
import torch
from config import *


train_loader, val_loader = get_data_loaders()
print('train loaders created')
model = get_deep_lab_v3()
print('model created')
criterion = torch.nn.MSELoss(reduction = 'mean')
optimizer = torch.optim.Adam(model.parameters(), lr = LR)
train_loss = train_one_epoch(model = model, criterion = criterion, optimizer = optimizer, dataLoader = train_loader)
print('one epoch trained created')
print(train_loss)
#pip install opencv-python==4.5.5.64

'''

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




'''




