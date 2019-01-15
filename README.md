# DCIC_2019_Glodon

[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)

---

### 快速开始：

- 首先在[百度网盘](https://pan.baidu.com/s/1wekoQM_TL1HWi3uxmDYkFw)，或者通过`wget https://pjreddie.com/media/files/yolov3.weights`下载yolov3的预训练文件,然后将下载好的模型文件放入**model_data**文件夹下

- 然后依次运行以下代码：

1. step_1_process_data.py　　　　将原本的csv文件转变成voc格式的xml文件

2. step_2_mv_data.py　　　　     移动文件到指定的目录(方便yolov3模型读写)

3. step_3_voc_annotation.py　　  自动构建训练目录

4. step_4_train.py　　　　        训练模型

5. step_5_predict.py　　　      　预测并生成最后提交结果


---

### 参数修改：

- anchors 大小在 model_data 下 yolo_anchors.txt
