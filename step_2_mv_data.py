# 移动文件
# 将训练集和测试集照片全部放到一起
import shutil
import os

train_jpg = os.listdir('data/train_dataset')
test_jpg = os.listdir('data/test_dataset')
train_xml = os.listdir('data/train_xml')

if not os.path.exists('VOCdevkit/VOC2007/ImageSets/Main'):
    os.mkdir('VOCdevkit/VOC2007/ImageSets/Main')

if not os.path.exists('VOCdevkit/VOC2007/JPEGImages'):
    os.mkdir('VOCdevkit/VOC2007/JPEGImages')

if not os.path.exists('VOCdevkit/VOC2007/Annotations'):
    os.mkdir('VOCdevkit/VOC2007/Annotations')

f = open('VOCdevkit/VOC2007/ImageSets/Main/train.txt', 'w')
for i in train_jpg:
    shutil.copy('data/train_dataset/' + i, 'VOCdevkit/VOC2007/JPEGImages/' + i)
    f.write(i.split('.')[-2])
    f.write('\n')
f.close()

f = open('VOCdevkit/VOC2007/ImageSets/Main/test.txt', 'w')
for i in test_jpg:
    shutil.copy('data/test_dataset/' + i, 'VOCdevkit/VOC2007/JPEGImages/' + i)
    f.write(i.split('.')[-2])
    f.write('\n')
f.close()

for i in train_xml:
    shutil.copy('data/train_xml/' + i, 'VOCdevkit/VOC2007/Annotations/' + i)
