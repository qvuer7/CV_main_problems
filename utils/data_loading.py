import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import albumentations as A
from torchvision import transforms



class segmentationDataset(Dataset):
    def __init__(self, dataFrame, transform):
        super().__init__()
        self.dataFrame = dataFrame
        self.transform = transform

    def __len__(self):
        return len(self.dataFrame)

    def __getitem__(self, x):
        image_path, mask_path = self.dataFrame.iloc[x].images, self.dataFrame.iloc[x].masks
        print(image_path)
        print(mask_path)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = np.where(mask == 0, 0, 1)
        mask = np.expand_dims(mask, axis=-1)

        if self.transform:
            data = self.transform(image=image, mask=mask)
            image = data['image']
            mask = data['mask']

        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        mask = np.transpose(mask, (2, 0, 1)).astype(np.float32)
        image = torch.Tensor(image)
        mask = torch.Tensor(mask)
        image = image / 255.0

        return image, mask


def test():
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
        A.Resize(image_heigh, image_width),
    ])

    i = 25
    train_ds = segmentationDataset(train_df, train_transform)
    image, mask = train_ds[i]
    image_ori = cv2.imread(train_df['images'][i])
    image_ori = cv2.cvtColor(image_ori, cv2.COLOR_BGR2RGB)
    mask_ori = cv2.imread(train_df['masks'][i])
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    image = np.array(image)
    image = np.transpose(image, (1, 2, 0))
    mask = np.array(mask)
    mask = np.transpose(mask, (1, 2, 0))
    ax1.imshow(image)
    ax2.imshow(mask)
    ax3.imshow(image_ori)
    ax4.imshow(mask_ori)
    plt.show()



if __name__ == '__main__':
    test()