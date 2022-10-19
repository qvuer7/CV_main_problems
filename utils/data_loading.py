import os
import xml.etree.ElementTree as et

import albumentations as A
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from utils.models import *
from utils.vizualisation import *


# ----------------------------------------------------------------------
#
#
#                       DETECTION DATA UTILS
#
#
# ----------------------------------------------------------------------


class DetectionDataset(Dataset):
    def __init__(self, dataFrame, labels_map, transform=None):
        super().__init__()
        self.dataFrame = dataFrame
        self.transform = transform
        self.labels_map = labels_map

    def __len__(self):
        return len(self.dataFrame)

    def __getitem__(self, x):

        image_path = self.dataFrame.iloc[x].image
        label_path = self.dataFrame.iloc[x].label

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = image / 255.0
        tree = et.parse(label_path)
        root = tree.getroot()
        parsed_labels = []
        data = []
        for member in root.findall('object'):
            parsed_labels.append(self.labels_map[member.find('name').text])  # label of object
            xmin = int(member.find('bndbox').find('xmin').text)  # coordinates of boxes pascal_voc format
            xmax = int(member.find('bndbox').find('xmax').text)  # coordinates of boxes pascal_voc format
            ymin = int(member.find('bndbox').find('ymin').text)  # coordinates of boxes pascal_voc format
            ymax = int(member.find('bndbox').find('ymax').text)  # coordinates of boxes pascal_voc format
            box = [xmin, ymin, xmax, ymax]  # pure coordinates array
            box.append(parsed_labels[0])  # coordinates with label for transformation(albumentations requirement)
            data.append(box)  # coordinates with label for transformation(albumentations requirement)

        if self.transform:
            out = self.transform(image=image, bboxes=data)
            image = out['image']
            box = out['bboxes']

        boxes = get_boxes_after_transformations(box)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        labels = torch.as_tensor(parsed_labels, dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['area'] = area
        target['iscrowd'] = iscrowd
        target['labels'] = labels

        image = image.transpose((2, 0, 1))
        image = torch.Tensor(image)

        return image, target


def get_boxes_after_transformations(boxes):
    out = []
    for i in range(len(boxes)):
        out.append(list(map(lambda x: int(x), boxes[i][:4])))

    return out


def get_detection_transforms():
    train_transforms = A.Compose([
        A.Resize(IMAGE_HEIGH, IMAGE_WIDTH),
        A.Flip(p=0.5),
        A.RandomRotate90(p=1),
        # ToTensorV2(p=1.0),
    ], bbox_params=A.BboxParams(format='pascal_voc'))

    test_transforms = A.Compose([
        A.Resize(IMAGE_HEIGH, IMAGE_WIDTH)
    ], bbox_params=A.BboxParams(format='pascal_voc'))

    return train_transforms, test_transforms


def collate_fn(batch):
    """
    To handle the data loading as different images may have different number
    of objects and to handle varying size tensors as well.
    """
    return tuple(zip(*batch))


def get_detection_data_loaders():
    train_df, test_df = get_dataframes()
    train_transform, test_transform = get_detection_transforms()
    train_dataset = DetectionDataset(dataFrame=train_df, labels_map=LABELS_MAP, transform=train_transform)
    test_dataset = DetectionDataset(dataFrame=test_df, labels_map=LABELS_MAP, transform=test_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    return train_loader, test_loader


def test_detection():
    i = 200
    train_df, test_df = get_dataframes()
    train_transform, test_transform = get_detection_transforms()

    train_loader, test_loader = get_detection_data_loaders()

    for i, data in enumerate(train_loader):
        pass
    print('data loaders tested fine')

    train_loader, test_loader = get_detection_data_loaders()
    train_dataset = DetectionDataset(dataFrame=train_df, labels_map=LABELS_MAP, transform=train_transform)
    test_dataset = DetectionDataset(dataFrame=test_df, labels_map=LABELS_MAP, transform=test_transform)
    image, target = train_dataset[i]

    image = draw_bounding_box_from_ITtensor(image, target, LABELS_MAP)
    cv2.imshow('iamge', image)

    for i, (image, target) in enumerate(train_loader):
        break
    print(image[0].shape)
    print(target)
    cv2.waitKey(0)


# ----------------------------------------------------------------------
#
#
#                       SEGMENTATION DATA UTILS
#
#
# ----------------------------------------------------------------------
class segmentationDataset(Dataset):
    def __init__(self, dataFrame, transform):
        super().__init__()
        self.dataFrame = dataFrame
        self.transform = transform

    def __len__(self):
        return len(self.dataFrame)

    def __getitem__(self, x):
        image_path, mask_path = self.dataFrame.iloc[x].image, self.dataFrame.iloc[x].label
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


def get_segmentation_data_loaders():
    train_df, test_df = get_dataframes()
    train_transform, test_transform = get_segmentation_transforms()
    train_ds = segmentationDataset(train_df, train_transform)
    test_ds = segmentationDataset(test_df, test_transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    return train_loader, test_loader


def get_segmentation_transforms():
    train_transform = A.Compose([
        A.LongestMaxSize(IMAGE_HEIGH),
        A.HorizontalFlip(p=0.5)

    ])

    test_transform = A.Compose([
        A.LongestMaxSize(IMAGE_HEIGH)
    ])

    return train_transform, test_transform


def test_segmentation():
    train_df, test_df = get_dataframes()
    train_transform, test_transform = get_segmentation_transforms()
    train_ds = segmentationDataset(train_df, train_transform)
    i = 12
    image, mask = train_ds[i]
    image_ori = cv2.imread(train_df['image'][i])
    image_ori = cv2.cvtColor(image_ori, cv2.COLOR_BGR2RGB)
    mask_ori = cv2.imread(train_df['label'][i])
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
    train_loader, test_loader = get_segmentation_data_loaders()

    for i, (image, mask) in train_loader:
        break
    print(image.shape)
    print(mask.shape)


def inference_deep_lab(image_m, checkpoint_path, threshold):
    ''' EXAMPLE
    train_df, test_df = get_dataframes()
    _, test_transforms = get_segmentation_transforms()
    dataset = segmentationDataset(dataFrame = test_df, transform = test_transforms)
    image, mask = dataset[1]
    output_mask = inference_deep_lab(image_m = image, checkpoint_path = 'custom/checkpoint/path, threshold = custom.threshold)
    vizualize_segmentation_output(image_ori = image, mask_ori = mask, mask = output_mask)
    '''

    if threshold > 1:
        threshold /= 100
    image = image_m.clone().detach()
    image = image.unsqueeze(0)
    image = image.to(DEVICE)
    model, _ = get_deep_lab_v3()
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    output = model(image)["out"][0]
    output = torch.sigmoid(output)
    output = (output > threshold) * 1.0
    output = output.cpu().numpy()
    output = np.where(output[0] == 1, 1, 0)
    return output


# ----------------------------------------------------------------------
#
#
#                       GENERAL FUNCTIONS
#
#
# ----------------------------------------------------------------------

def get_dataframes():
    annotations = os.listdir(ANNOTATIONS_PATH)
    images = os.listdir(IMAGES_PATH)
    images_full = list(map(lambda x: IMAGES_PATH + x, images))
    annotations_full = list(map(lambda x: ANNOTATIONS_PATH + x, annotations))
    images_full.sort()
    annotations_full.sort()
    dataset_df = pd.DataFrame({'image': images_full, 'label': annotations_full})
    train_df, test_df = train_test_split(dataset_df, train_size=TRAIN_RATIO, shuffle=False)
    return train_df, test_df


if __name__ == '__main__':
    if TASK == 1:

        test_segmentation()
    elif TASK == 2:
        test_detection()
