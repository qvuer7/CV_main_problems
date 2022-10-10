from tqdm import tqdm
from config import *
import numpy as np

def train_one_epoch(model, criterion, optimizer, dataLoader):
    model.train()
    total_loss = 0
    for i, (image, mask) in enumerate(tqdm(dataLoader)):
        image = image.to(DEVICE)
        mask = mask.to(DEVICE)
        optimizer.zero_grad()
        out = model(image)
        loss = criterion(out['out'], mask)
        total_loss+=loss.item()
        loss.backward()
        optimizer.step()
    return total_loss/(i*BATCH_SIZE)

def val_one_epoch(model,criterion, dataLoader):
    model.eval()
    total_loss = 0
    for i, (image, mask) in enumerate(tqdm(dataLoader)):
        image = image.to(DEVICE)
        mask = mask.to(DEVICE)
        out = model(image)
        loss = criterion(out['out'], mask)
        total_loss+=loss.item()

    return total_loss/(i*BATCH_SIZE)


def train(model, criterion,optimizer, trainDataLoader, testDataLoader):
    best_loss = np.inf
    for epoch in range(NUM_EPOCHS):
        train_loss =train_one_epoch(model, criterion, optimizer, trainDataLoader)
        v
