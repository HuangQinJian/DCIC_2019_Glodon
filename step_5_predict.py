from yolo import YOLO
from PIL import Image
from tqdm import *
import pandas as pd
import numpy as np
import os
import time
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5,6,7"

if not os.path.exists('results_imgs'):
    os.mkdir('results_imgs')

if not os.path.exists('result'):
    os.mkdir('result')

# 修改的参数

yolo = YOLO()
f = open('VOCdevkit/VOC2007/ImageSets/Main/test.txt')
text = f.read()
text_list = text.split('\n')
del text_list[-1]

jpg_list = []
xmin_list = []
ymin_list = []
xmax_list = []
ymax_list = []


def get_class(classes_path):
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


for i in range(len(text_list)):
    start = time.time()
    print('开始检测第'+str(i+1)+'张图片。')
    image_path = 'VOCdevkit/VOC2007/JPEGImages/' + text_list[i] + '.jpg'
    image = Image.open(image_path)
    image, label_record, score_record, top_record, left_record, bottom_record, right_record = yolo.detect_image(
        image)
    image.save('results_imgs/{}.png'.format(text_list[i]))
    all_classes = get_class('model_data/my_classes.txt')
    xmin_list.extend(left_record)
    ymin_list.extend(top_record)
    xmax_list.extend(right_record)
    ymax_list.extend(bottom_record)
    jpg = [text_list[i] + '.jpg'] * len(label_record)
    jpg_list.extend(jpg)
    print('第'+str(i+1)+'张图片检测完毕,检测出了'+str(len(label_record))+'个物体。')
    end = time.time()
    print('检测总共花费的时间为: {0:.2f}s'.format(end - start))


result = pd.DataFrame()
name_list = []
data_list = []
for i, value in enumerate(tqdm(jpg_list)):
    name_list.append(value)
    data_list.append(str(xmin_list[i]) + ' ' + str(ymin_list[i]) +
                     ' ' + str(xmax_list[i]) + ' ' + str(ymax_list[i]))

result['id'] = name_list
result['pos'] = data_list
result.to_csv('result/baseline.csv', index=False, header=None)
