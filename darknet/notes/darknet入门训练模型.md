
已实践了几次,正确性进一步尚待验证.简单说明:



# 如果看到avg loss =nan

说明训练错误; 某一行的Class=-nan说明目标太大或者太小,某个尺度检测不到,属于正常 

参考链接:

[使用caffe训练时Loss变为nan的原因](https://zhuanlan.zhihu.com/p/25110930)

[YOLOv3训练过程中重要参数的理解和输出参数的含义](https://blog.csdn.net/maweifei/article/details/81148414)

# 什么时候应该停止训练?

当loss不在下降或者下降极慢的情况可以停止训练,一般loss=0.7左右就可以了

# 在训练集上测试正确率很高,在其他测试集上测试效果很差？

说明过拟合了.解决,提前停止训练,或者增大样本数量训练

# 如何提高目标检测正确率包括IOU,分类正确率？

设置yolo层 random =1,增加不同的分辨率.或者增大图片本身分辨率.或者根据你自定义的数据集去重新计算anchor尺寸
```
darknet.exe detector calc_anchors data/obj.data -num_of_clusters 9 -width 416 -height 416 then set the same 9 anchors in each of 3 [yolo]-layers in your cfg-file
```

# 如何增加训练样本？

样本特点尽量多样化,亮度,旋转,背景,目标位置,尺寸.添加没有标注框的图片和其空的txt文件,作为negative数据.

# 训练的图片较小,但是实际检测图片大,怎么检测小目标？     

1. 使在用416*416训练完之后,也可以在cfg文件中设置较大的width和height,增加网络对图像的分辨率,从而更可能检测出图像中的小目标,而不需要重新训练
    
2. set `[route] layers = -1, 11` set ` [upsample] stride=4`

# 网络模型耗费资源多大？（我用过就两个）

    [yolov3.cfg]  [236MB COCO-91类]  [4GB GPU-RAM]
    [yolov3.cfg]  [194MB VOC-20类]  [4GB GPU-RAM]
    [yolov3-tiny.cfg]  [34MB COCO-91类]  [1GB GPU-RAM]
    [yolov3-tiny.cfg]  [26MB VOC-20类]  [1GB GPU-RAM]
    
# 多GPU怎么训练？

首先用一个gpu训练1000次迭代后的网络,再用多gpu训练
```
darknet.exe detector train data/voc.data cfg/yolov3-voc.cfg /backup/yolov3-voc_1000.weights -gpus 0,1,2,3  
```

# 有哪些命令行来对神经网络进行训练和测试？

1. 检测图片: 
```
build\darknet\x64\darknet.exe detector test data/coco.data cfg/yolov3.cfg yolov3.weights  -thresh 0.25 xxx.jpg
```
2. 检测视频:将test 改为 demo ; xxx.jpg 改为xxx.mp4

3. 调用网络摄像头:将xxx.mp4 改为 http://192.168.0.80:8080/video?dummy=x.mjpg -i 0

4. 批量检测:-dont_show -ext_output < data/train.txt >  result.txt

5. 手持端网络摄像头:下载mjpeg-stream 软件, xxx.jpg 改为 IP Webcam / Smart WebCam

# 如何评价模型好坏？
```
build\darknet\x64\darknet.exe detector map data\defect.data cfg\yolov3.cfg backup\yolov3.weights
```
利用上面命令计算各权重文件,选择具有最高IoU（联合的交集）和mAP（平均精度）的权重文件
# 如果测试时出现权重不能识别目标的情况？

很可能是因为yolov3.cfg文件开头的Traing下面两行没有注释掉,并且把Testing下面两行去掉注释就可以正常使用.
# 模型什么时候保存？如何更改？
迭代次数小于1000时,每100次保存一次,大于1000时,没10000次保存一次.自己可以根据需求进行更改,然后重新编译即可[ 先 make clean ,然后再 make].

代码位置: examples/detector.c line 138


# darknet特点
优点:速度快,精度提升,小目标检测有改善；

不足:中大目标有一定程度的削弱,遮挡漏检,速度稍慢于V2.

