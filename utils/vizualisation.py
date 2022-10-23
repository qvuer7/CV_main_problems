import cv2
import numpy as np
import matplotlib.pyplot as plt
from config import *


def draw_bounding_box_from_albumentations(image, box):
    # image -> numpy | box -> list of tumples?
    for i in range(len(box)):
        print(i)
        boxes = list(map(lambda x: int(x), box[i][:4]))
        label = box[i][4]

        image = cv2.rectangle(image, (boxes[0], boxes[1]), (boxes[2], boxes[3]), (255, 0, 0), 2)

    return image


def draw_bounding_box_from_Ttensor(image, target, label_map):
    box = np.asarray(target['boxes'], dtype=np.int32)
    labels = np.asarray(target['labels'], dtype=np.int32)
    colors = {0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255)}
    decoding_label_map = {value: key for (key, value) in label_map.items()}
    for i, b in enumerate(box):
        image = cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), colors[labels[i]], 2)
        image = cv2.putText(image, decoding_label_map[labels[i]], (b[0], b[1]), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            colors[labels[i]], 2, cv2.LINE_AA)

    return image


def draw_bounding_box_from_ITtensor(image_s, target, label_map):
    image = image_s.clone()
    image = np.asarray(image)
    image = image.transpose((1, 2, 0))
    image = draw_bounding_box_from_Ttensor(image=image, target=target, label_map=label_map)
    return image


def vizualize_segmentation_output(image_ori, mask_ori, mask):
    image = image_ori.clone()
    image = image.numpy()
    image = image.transpose((1, 2, 0))
    image_to_draw_output = image.copy()
    image_to_draw_original = image.copy()
    mask_ori = mask_ori.numpy().squeeze()
    m1 = np.where(mask_ori == 1)
    m2 = np.where(mask == 1)

    image_to_draw_output[m2[0], m2[1], 0] = 70
    image_to_draw_original[m1[0], m1[1], 0] = 70
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(12, 10))
    fig.suptitle('Segmentation network inference visualisation', fontsize=16)
    ax1.imshow(image_to_draw_original)
    ax2.imshow(image_to_draw_output)
    ax3.imshow(mask_ori)
    ax4.imshow(mask)


    ax1.set_title('original image')
    ax2.set_title('output image w mask')
    ax3.set_title('original mask')
    ax4.set_title('output mask')
    plt.show()



def draw_instance_inference_and_original(original_image, original_target, out_target):

    '''
    checkpoint_path = '/Users/andriizelenko/qvuer7/projects/CV_main_tasks/checkpoints/instance_segmentation/checkpoint_instance_segmentation.pth'
    train_df, test_df = get_instance_segmentation_dataframes()
    train_t, test_t = get_detection_transforms()
    md = InstanceSegmentationDataset(test_df, transforms = test_t, labels_map = LABELS_MAP)
    i = 12
    image_m, target = md[i]
    '''

    out_target['masks'] = out_target['masks'].squeeze(1)
    masks = np.asarray(out_target['masks'])
    mask_output = 0
    for mask in masks:
        mask = np.array(mask)
        mask_output+= mask

    mask_original = 0
    for mask in original_target['masks']:
        mask = np.array(mask)
        mask_original+= mask
    image_model_w_boxes = draw_bounding_box_from_ITtensor(image_s = original_image, target = out_target, label_map = LABELS_MAP)
    image_original_w_boxes = draw_bounding_box_from_ITtensor(image_s = original_image, target = original_target, label_map = LABELS_MAP)
    fig, ((ax1, ax2), (ax3,ax4)) = plt.subplots(2,2, figsize = (14,10))
    ax1.imshow(image_original_w_boxes)
    ax2.imshow(image_model_w_boxes)
    ax3.imshow(mask_original)
    ax4.imshow(mask_output)
    fig.suptitle('Instance segmentation network inference visualisation', fontsize=16)
    ax1.set_title('Original image w boxes')
    ax2.set_title('Original image w output boxes')
    ax3.set_title('Original mask')
    ax4.set_title('Output mask')
    plt.show()



def draw_detection_output(image_ori, output, target):

    image = draw_bounding_box_from_ITtensor(image_s = image_ori, target = output, label_map=LABELS_MAP)
    image_2 = draw_bounding_box_from_ITtensor(image_s = image_ori, target = target, label_map = LABELS_MAP)
    fig, (ax1, ax2) = plt.subplots(1,2)
    fig.suptitle('Detection network inference visualisation', fontsize=16)
    ax1.imshow(image_2)
    ax1.set_title('original image')
    ax2.imshow(image)
    ax2.set_title('output from detector')
    plt.show()
