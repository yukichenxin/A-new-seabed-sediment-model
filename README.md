## A Precise Semantic Segmentation Model for Seabed Sediment Detection using YOLO-C


## Top News
** It supports multi-GPU training for faster convergence and improved efficiency. The model incorporates a novel object quantity calculation and heatmap generation.**  

**It also supports popular learning rate scheduling techniques and allows for flexible optimization algorithm selection. YOLO-C includes adaptive learning rate adjustment based on batch size and enables image cropping. **  

**It supports training of models with different sizes and provides a wide range of adjustable parameters. Additionally, YOLO-C offers functionalities such as real-time object detection, video prediction, and batch inference. These features collectively contribute to the versatility and high performance of YOLO-C in various computer vision applications.**   

## 实现的内容
- [x] Based on the YOLO detection architecture
- [x] Utilizes the YOLO backbone and PAFPN as the encoder to achieve a UNet-like segmentation structure
- [x] Efficient fusion of the detection and segmentation structures

## 所需环境
pytorch==1.2.0


## Training step
In this study, the training was conducted using the VOC format. Prior to training, it is necessary to prepare the dataset as follows:
1. Put the label files in the "Annotation" folder, which is located under the "VOC2007" folder in the "VOCdevkit" directory.
2. Put the image files in the "JPEGImages" folder, which is located under the "VOC2007" folder in the "VOCdevkit" directory.
3. Put the image labels in the "SegmentationClass" folder, which is located under the "VOC2007" folder in the "VOCdevkit" directory.
These steps ensure that the dataset is properly organized and ready for training.
```python
CUDA_VISIBLE_DEVICES=0 python train.py
```

## Predict 
**modify root yolo.py**
```python
CUDA_VISIBLE_DEVICES=0 python predict.py
```

## Evaluate 
**modify root get_miou.py和get_map.py**
```python
#Get the metric of segmentation
CUDA_VISIBLE_DEVICES=0 python get_miou.py 
```

```python
#Get the metric of detection
CUDA_VISIBLE_DEVICES=0 python get_map.py 
```
