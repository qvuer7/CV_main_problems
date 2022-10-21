from tqdm import tqdm
from config import *
import numpy as np




#------------------------------------------------------------------------#
#                                                                        #
#                                                                        #
#                   INSTANCE SEGMENTATION TRAINERS                       #
#                                                                        #
#                                                                        #
#------------------------------------------------------------------------#


def train_one_epoch_instance(dataLoader, model, optimizer):
    model.train()
    total_loss = 0
    for i, (images, targets) in enumerate(tqdm(dataLoader)):
        images = list(image.to(DEVICE) for image in images)
        targets = [{k:v.to(DEVICE) for k, v in t.items()} for t in targets]
        optimizer.zero_grad()
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        total_loss += losses.item() * BATCH_SIZE
        losses.backward()
        optimizer.step()

    return total_loss/i


def validate_one_epoch_instance(dataLoader, model):
    total_loss = 0
    for i, (images, targets) in enumerate(tqdm(dataLoader)):
        images = list(image.to(DEVICE) for image in images)
        targets = [{k:v.to(DEVICE) for k,v in t.items()} for t in targets]
        with torch.no_grad():
            loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        total_loss += losses.item() * BATCH_SIZE
    return total_loss / i

def train_instance(model, trainLoader, testLoader, optimizer):
    best_val_loss = torch.inf
    model.to(DEVICE)
    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch_instance(trainLoader, model, optimizer)
        val_loss = validate_one_epoch_instance(testLoader, model)
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/validation", val_loss, epoch)
        print('-'*10 + f'epoch {epoch}'+'-'*10)
        print(f'training loss  :    {train_loss}')
        print(f'validation loss:    {val_loss}')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CHECKPOINTS_PATH + 'checkpoint_instance_segmentation.pth')
            print('model_saved')



#------------------------------------------------------------------------#
#                                                                        #
#                                                                        #
#                   DETECTION    TRAINERS                                #
#                                                                        #
#                                                                        #
#------------------------------------------------------------------------#

def train_one_epoch_detection(model, dataLoader, optimizer):
    model.train()
    total_loss = 0
    for i, data in enumerate(tqdm(dataLoader)):
        images, targets = data
        images = list(image.to(DEVICE) for image in images)
        targets = [{k:v.to(DEVICE) for k, v in t.items()} for t in targets]
        optimizer.zero_grad()
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        total_loss+= losses.item()*BATCH_SIZE
        losses.backward()
        optimizer.step()

    return total_loss/i

def validate_one_epoch_detection(model, dataLoader):
    total_loss = 0
    for i, data in enumerate(tqdm(dataLoader)):
        images, targets = data
        images = list(image.to(DEVICE) for image in images)
        targets = [{k:v.to(DEVICE) for k,v in t.items()} for t in targets]
        with torch.no_grad():
            loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        total_loss+=losses.item()*BATCH_SIZE
    return total_loss/i


def train_detector(model, trainLoader, testLoader, optimizer):
    best_val_loss = torch.inf
    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch_detection(model = model, dataLoader = trainLoader, optimizer=optimizer)
        val_loss = validate_one_epoch_detection(model = model, dataLoader = testLoader)
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/validation", val_loss, epoch)
        print('-'*10 + f'epoch {epoch}'+'-'*10)
        print(f'training loss  :    {train_loss}')
        print(f'validation loss:    {val_loss}')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CHECKPOINTS_PATH + 'checkpoint_detector.pth')
            print('model_saved')
    writer.flush()


#------------------------------------------------------------------------#
#                                                                        #
#                                                                        #
#                   SEGMENTATION TRAINERS                                #
#                                                                        #
#                                                                        #
#------------------------------------------------------------------------#
def train_one_epoch_segmentation(model, criterion, optimizer, dataLoader):
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

    return total_loss/i

def val_one_epoch_segmentation(model,criterion, dataLoader):
    model.eval()
    total_loss = 0
    for i, (image, mask) in enumerate(tqdm(dataLoader)):
        image = image.to(DEVICE)
        mask = mask.to(DEVICE)
        out = model(image)
        loss = criterion(out['out'], mask)
        total_loss+=loss.item()

    return total_loss/i


def train_segmentation(model, criterion,optimizer, trainDataLoader, testDataLoader):
    best_loss = np.inf
    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch_segmentation(model, criterion, optimizer, trainDataLoader)
        val_loss = val_one_epoch_segmentation(model, criterion, testDataLoader)
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/validation", val_loss, epoch)
        print('-'*10 + f'epoch {epoch}'+'-'*10)
        print(f'training loss  :    {train_loss}')
        print(f'validation loss:    {val_loss}')
        if best_loss > val_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), CHECKPOINTS_PATH + 'checkpoint_segmenter' '.pth')

    writer.flush()
