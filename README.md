## YOLOC 联合语义分割和目标检测的高效融合算法


## Top News
**支持多GPU训练，新增各个种类目标数量计算，新增heatmap。**  

**支持step、cos学习率下降法、支持adam、sgd优化器选择、支持学习率根据batch_size自适应调整、新增图片裁剪。**  

**支持不同尺寸模型训练、支持大量可调整参数，支持fps、视频预测、批量预测等功能。**   

## 实现的内容
- [x] 基于YOLO的检测结构
- [x] 基于YOLO的backbone和PAFPN作为Encoder，实现类unet结构的分割结构
- [x] 检测结构和分割结构的高效融合

## 所需环境
pytorch==1.2.0


## 训练步骤
**本文使用VOC格式进行训练，训练前需要自己制作好数据集，**    
训练前将标签文件放在VOCdevkit文件夹下的VOC2007文件夹下的Annotation中。   
训练前将图片文件放在VOCdevkit文件夹下的VOC2007文件夹下的JPEGImages中。
训练前将图片标签放在VOCdevkit文件夹下的VOC2007文件夹下的SegmentationClass中。
```python
CUDA_VISIBLE_DEVICES=0 python train.py
```

## 预测步骤
**修改好根目录下的yolo.py**
```python
CUDA_VISIBLE_DEVICES=0 python predict.py
```

## 评估步骤 
**修改好根目录下的get_miou.py和get_map.py**
```python
#得到分割的指标
CUDA_VISIBLE_DEVICES=0 python get_miou.py 
```

```python
#得到检测的指标
CUDA_VISIBLE_DEVICES=0 python get_map.py 
```
