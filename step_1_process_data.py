# 生成xml标注文件
import pandas as pd
from PIL import Image
import os

data = pd.read_csv('data/train_labels.csv')
del data['AB']
data['temp'] = data['ID']

if not os.path.exists('data/train_xml'):
    os.mkdir('data/train_xml/')


def save_xml(image_name, name_list, xmin_list, ymin_list, xmax_list, ymax_list):
    xml_file = open('data/train_xml/' +
                    image_name.split('.')[-2] + '.xml', 'w')
    image_name = 'data/train_dataset/' + image_name
    img = Image.open(image_name)
    img_width = img.size[0]
    img_height = img.size[1]
    xml_file.write('<annotation>\n')
    xml_file.write('    <folder>' + image_name.split('/')[-2] + '</folder>\n')
    xml_file.write('    <filename>' + image_name.split('/')
                   [-1] + '</filename>\n')
    xml_file.write('    <path>' + image_name + '</path>\n')
    xml_file.write('    <source>\n')
    xml_file.write('        <database>Unknown</database>\n')
    xml_file.write('    </source>\n')
    xml_file.write('    <size>\n')
    xml_file.write('        <width>' + str(img_width) + '</width>\n')
    xml_file.write('        <height>' + str(img_height) + '</height>\n')
    xml_file.write('        <depth>3</depth>\n')
    xml_file.write('    </size>\n')
    xml_file.write('    <segmented>0</segmented>\n')

    data_panda = pd.DataFrame()
    data_panda['name_list'] = name_list
    data_panda['xmin_list'] = xmin_list
    data_panda['ymin_list'] = ymin_list
    data_panda['xmax_list'] = xmax_list
    data_panda['ymax_list'] = ymax_list

    new_panda = data_panda.drop_duplicates(
        ['name_list', 'xmin_list', 'ymin_list', 'xmax_list',   'ymax_list'], keep='first')
    name_list = list(new_panda['name_list'])
    xmin_list = list(new_panda['xmin_list'])
    ymin_list = list(new_panda['ymin_list'])
    xmax_list = list(new_panda['xmax_list'])
    ymax_list = list(new_panda['ymax_list'])

    if len(data_panda) != len(new_panda):
        print(data_panda)
        print(new_panda)

    for i, value in enumerate(name_list):
        xml_file.write('    <object>\n')
        xml_file.write('        <name>' + name_list[i] + '</name>\n')
        xml_file.write('        <pose>Unspecified</pose>\n')
        xml_file.write('        <truncated>0</truncated>\n')
        xml_file.write('        <difficult>0</difficult>\n')
        xml_file.write('        <bndbox>\n')
        xml_file.write('            <xmin>' + str(xmin_list[i]) + '</xmin>\n')
        xml_file.write('            <ymin>' + str(ymin_list[i]) + '</ymin>\n')
        xml_file.write('            <xmax>' + str(xmax_list[i]) + '</xmax>\n')
        xml_file.write('            <ymax>' + str(ymax_list[i]) + '</ymax>\n')
        xml_file.write('        </bndbox>\n')
        xml_file.write('    </object>\n')
    xml_file.write('</annotation>')
    xml_file.close()


def dealed_detection(row):
    detection = list(row['Detection'])
    xmin_list = []
    ymin_list = []
    xmax_list = []
    ymax_list = []
    name_list = []
    for i in detection:
        temp = i.split(' ')
        xmin_list.append(temp[0])
        ymin_list.append(temp[1])
        xmax_list.append(temp[2])
        ymax_list.append(temp[3])
        name_list.append('AB')
    save_xml(list(row['temp'])[0], name_list,
             xmin_list, ymin_list, xmax_list, ymax_list)


data.groupby('ID').apply(lambda row: dealed_detection(row))
