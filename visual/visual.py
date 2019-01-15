import matplotlib.pyplot as plt
from skimage.io import imread, imshow

import os
import cv2
import numpy as np
import pandas as pd


def get_boxes(img_boxes):
    boxes_list = []
    for i in range(len(img_boxes)):
        box_list = []
        axes = img_boxes[i]
        x1 = int(axes.split(' ')[0])
        y1 = int(axes.split(' ')[1])
        x2 = int(axes.split(' ')[2])
        y2 = int(axes.split(' ')[3])
        box_list.append(x1)
        box_list.append(y1)
        box_list.append(x2)
        box_list.append(y2)
        boxes_list.append(box_list)
    return boxes_list


def draw_rectangle(boxes, image, is_yolo_out=False):
    new_box = np.array(boxes)
    print(new_box)
    for box in new_box:
        if is_yolo_out:
            left = np.rint(box[0] - box[2] / 2)
            right = np.rint(box[1] - box[3] / 2)
            top = np.rint(box[0] + box[2] / 2)
            bottom = np.rint(box[1] + box[3] / 2)
        else:
            left = np.rint(box[0])
            right = np.rint(box[1])
            top = np.rint(box[2])
            bottom = np.rint(box[3])
        print('left, right, top, bottom:', left, right, top, bottom)
        cv2.rectangle(image,
                      (int(left), int(right)),
                      (int(top), int(bottom)),
                      (0, 255, 0),
                      1)

    cv2.namedWindow("Image")
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    img_path = '../VOCdevkit/VOC2007/JPEGImages/'
    img_list = os.listdir(img_path)
    train = pd.read_csv('../data/train_labels.csv')
    for i in range(len(img_list)):
        img_name = img_list[i]
        img_raw = cv2.imread(os.path.join(img_path, img_name))
        img = train[train.ID == img_name]
        img_boxes = img['Detection']
        boxes_list = get_boxes(list(img_boxes))
        draw_rectangle(boxes_list, img_raw)
        break
