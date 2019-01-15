# keras-yolo3

[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)

---

# 首先在 https://pan.baidu.com/s/1p04_dYLgUmEUIhzIug_KGA 这里下载yolov3的预训练文件(将下载好的模型文件放入model_data文件夹下)
# 将下载好的数据放入data文件夹下

step_1_process_data.py 将原本的csv文件转变成voc格式的xml文件
step_2_mv_data.py 移动文件到指定的目录(方便yolov3模型读写)
step_3_voc_annotation.py 自动构建训练目录
step_4_train.py 训练模型(cpu上我训练了一天)
step_5_yolo_vodeo.py 预测并生成最后提交结果

分数比较低只有0.2左右，听群里说调anchors有效果
anchors大小在model_data下yolo_anchors.txt