'''
@Description:
@Author: HuangQinJian
@Date: 2019-01-10 13:25:54
@LastEditTime: 2019-01-18 14:03:20
@LastEditors: HuangQinJian
'''
import os
import cv2
import time
import random
import shutil
import numpy as np
import pandas as pd
from skimage import img_as_float
from visual.visual import get_boxes, draw_rectangle

if not os.path.exists('data/augument'):
    os.mkdir('data/augument')

augment_img_path = 'data/augument/'


def augment(img, boxes, augment_type):

    rows, cols = img.shape[:2]
    boxes_augment = []
    # 随机变换亮度
    if augment_type == 'random_bright':
        if random.random() < 0.8:
            alpha = random.uniform(0.3, 0.4)
            img = img.astype('float')
            img *= alpha
            img = img.clip(min=0, max=255)
            # print(img)
        boxes_augment = boxes

    if augment_type == 'horizontal_flips':
        img = cv2.flip(img, 1)
        for i in range(len(boxes)):
            bbox = boxes[i]
            box = []
            # print(bbox)
            box.append(cols-bbox[2])
            box.append(bbox[1])
            box.append(cols-bbox[0])
            box.append(bbox[3])
            # print(box)
            # break
            boxes_augment.append(box)

    if augment_type == 'vertical_flips':
        img = cv2.flip(img, 0)
        for i in range(len(boxes)):
            bbox = boxes[i]
            box = []
            # print(bbox)
            box.append(bbox[0])
            box.append(rows - bbox[1])
            box.append(bbox[2])
            box.append(rows - bbox[3])
            # print(box)
            # break
            boxes_augment.append(box)

    if augment_type == 'rotation':
        angle = np.random.choice([90, 180, 270], 1)[0]
        print(angle)
        if angle == 270:
            img = np.transpose(img, (1, 0, 2))
            img = cv2.flip(img, 0)
        elif angle == 180:
            img = cv2.flip(img, -1)
        elif angle == 90:
            img = np.transpose(img, (1, 0, 2))
            img = cv2.flip(img, 1)

        for i in range(len(boxes)):
            bbox = boxes[i]
            box = []
            # print(box)
            if angle == 270:
                box.append(bbox[1])
                box.append(cols - bbox[2])
                box.append(bbox[3])
                box.append(cols - bbox[0])
            elif angle == 180:
                box.append(cols - bbox[2])
                box.append(rows - bbox[3])
                box.append(cols - bbox[0])
                box.append(rows - bbox[1])
            elif angle == 90:
                box.append(rows - bbox[3])
                box.append(bbox[0])
                box.append(rows - bbox[1])
                box.append(bbox[2])
            boxes_augment.append(box)
    return boxes_augment, img


def apply_augment(augment_type, img_path, train_label_path):
    img_list = os.listdir(img_path)
    train = pd.read_csv(train_label_path)
    train_augument = pd.DataFrame()
    box_augment_list = []
    img_name_augment_list = []
    start_total = time.time()
    for i in range(len(img_list)):
        start = time.time()
        img_name = img_list[i]
        # print(img_name)
        print('开始处理第{}张图片!'.format(i+1))
        img_raw = cv2.imread(os.path.join(img_path, img_name))
        img = train[train.ID == img_name]
        img_boxes = img['Detection']
        boxes_list = get_boxes(list(img_boxes))
        boxes_augment, img_augment = augment(
            img_raw, boxes_list, augment_type)

        img_aug_name = img_name.split('.')[0]+'_augument_'+augment_type+'.jpg'
        cv2.imwrite('data/augument/{}'.format(
            img_aug_name), img_augment)
        # draw_rectangle(boxes_augment, img_augment)
        for j in range(len(boxes_augment)):
            img_name_augment_list.append(
                img_aug_name)
            box = str(boxes_augment[j][0])+' '+str(boxes_augment[j][1]) + \
                ' '+str(boxes_augment[j][2])+' '+str(boxes_augment[j][3])
            box_augment_list.append(box)
        print('第{}张图片处理完毕!'.format(i+1))
        end = time.time()
        print('处理单张图片花费的时间为: {0:.2f}s'.format(end - start))

    train_augument['ID'] = img_name_augment_list
    train_augument['Detection'] = box_augment_list

    end_total = time.time()
    print('处理全部图片花费的时间为: {0:.2f}s'.format(end_total - start_total))

    return train_augument


def merge_augment_raw(train_label_path, train_augument, img_path, augment_img_path):
    train = pd.read_csv(train_label_path)
    train = train[['ID', 'Detection']]
    train_merge = pd.concat([train, train_augument])
    train_merge.to_csv('data/train_merge_labels.csv', index=None)
    augment_img_list = os.listdir(augment_img_path)
    for i in augment_img_list:
        shutil.copy(augment_img_path + i,
                    img_path + i)


if __name__ == '__main__':
    img_path = 'data/train_dataset/'
    # img_path = 'data/a/'
    train_label_path = 'data/train_labels.csv'
    augment_type = 'vertical_flips'
    train_augument = apply_augment(
        augment_type, img_path, train_label_path)

    augment_type_2 = 'random_bright'
    train_augument_2 = apply_augment(
        augment_type_2, img_path, train_label_path)

    train = pd.read_csv(train_label_path)
    train = train[['ID', 'Detection']]
    train_merge = pd.concat([train, train_augument])
    train_merge = pd.concat([train_merge, train_augument_2])
    train_merge.to_csv('data/train_merge_labels.csv', index=None)
    augment_img_list = os.listdir(augment_img_path)
    for i in augment_img_list:
        shutil.copy(augment_img_path + i,
                    img_path + i)

    # merge_augment_raw(train_label_path, train_augument,
    #                   img_path, augment_img_path)
    # train_augument.to_csv('data/train_augument_labels.csv', index=None)
