import cv2
import numpy as np
import matplotlib.pyplot as plt


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


def draw_bounding_box_from_ITtensor(image, target, label_map):
    image = np.asarray(image)
    image = image.transpose((1, 2, 0))
    image = draw_bounding_box_from_Ttensor(image=image, target=target, label_map=label_map)
    return image


def vizualize_segmentation_output(image_ori, mask_ori, mask):
    image_ori = image_ori.numpy()
    image_ori = image_ori.transpose((1, 2, 0))
    mask_ori = mask_ori.numpy().squeeze()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 12))
    ax1.imshow(image_ori)
    ax2.imshow(mask_ori)
    ax3.imshow(mask)
    plt.show()
