# Multispectral-Object-Detection

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cross-modality-fusion-transformer-for/multispectral-object-detection-on-flir)](https://paperswithcode.com/sota/multispectral-object-detection-on-flir?p=cross-modality-fusion-transformer-for)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cross-modality-fusion-transformer-for/pedestrian-detection-on-llvip)](https://paperswithcode.com/sota/pedestrian-detection-on-llvip?p=cross-modality-fusion-transformer-for)

[![New](https://img.shields.io/badge/2021-NEW-brightgreen.svg)](https://github.com/DocF/multispectral-object-detection/)
![Visitors](https://visitor-badge.glitch.me/badge?page_id=DocF.multispectral-object-detection)
[![GitHub stars](https://img.shields.io/github/stars/DocF/multispectral-object-detection.svg?style=social&label=Stars)](https://github.com/DocF/multispectral-object-detection)


## Intro
Official Code for [Cross-Modality Fusion Transformer for Multispectral Object Detection](https://arxiv.org/abs/2111.00273).

Multispectral Object Detection with Transformer and Yolov5

## Abstract
Multispectral image pairs can provide the combined information, making object detection applications more reliable and robust in the open world. 
To fully exploit the different modalities, we present a simple yet effective cross-modality feature fusion approach, named Cross-Modality Fusion Transformer (CFT) in this paper. 
Unlike prior CNNs-based works, guided by the Transformer scheme, our network learns long-range dependencies and integrates global contextual information in the feature extraction stage. 
More importantly, by leveraging the self attention of the Transformer, the network can naturally carry out simultaneous intra-modality and inter-modality fusion, and robustly capture the latent interactions between RGB and Thermal domains, thereby significantly improving the performance of multispectral object detection. 
Extensive experiments and ablation studies on multiple datasets demonstrate that our approach is effective and achieves state-of-the-art detection performance. 
### Demo
**Night Scene**
<div align="left">
<img src="https://github.com/DocF/multispectral-object-detection/blob/main/video/demo1.gif" width="600">
</div>

**Day Scene**
<div align="left">
<img src="https://github.com/DocF/multispectral-object-detection/blob/main/video/demo.gif" width="600">
</div>

 
### Overview
<div align="left">
<img src="https://github.com/DocF/multispectral-object-detection/blob/main/cft.png" width="800">
</div>

## Installation 
Python>=3.6.0 is required with all requirements.txt installed including PyTorch>=1.7 (The same as yolov5 https://github.com/ultralytics/yolov5 ).

#### Clone the repo
    git clone https://github.com/DocF/multispectral-object-detection
  
#### Install requirements
 ```bash
$ pip install -r requirements.txt
```

## Dataset
-[FLIR]  [[Google Drive]](http://shorturl.at/ahAY4) [[Baidu Drive]](https://pan.baidu.com/s/1z2GHVD3WVlGsVzBR1ajSrQ?pwd=qwer) ```extraction code:qwer``` 

  A new aligned version.

-[LLVIP]  [download](https://github.com/bupt-ai-cz/LLVIP)

-[VEDAI]  [download](https://downloads.greyc.fr/vedai/)


You need to convert all annotations to YOLOv5 format.

Refer: https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data

## Run
#### Download the pretrained weights
yolov5 weights (pre-train) 

-[yolov5s] [google drive](https://drive.google.com/file/d/1UGAsaOvV7jVrk0RvFVYL6Vq0K7NQLD8H/view?usp=sharing)

-[yolov5m] [google drive](https://drive.google.com/file/d/1qB7L2vtlGppGjHp5xpXCKw14YHhbV4s1/view?usp=sharing)

-[yolov5l] [google drive](https://drive.google.com/file/d/12OFGLF73CqTgOCMJAycZ8lB4eW19D0nb/view?usp=sharing)

-[yolov5x] [google drive](https://drive.google.com/file/d/1e9xiQImx84KFQ_a7XXpn608I3rhRmKEn/view?usp=sharing)

CFT weights 

-[LLVIP] [google drive](https://drive.google.com/file/d/18yLDUOxNXQ17oypQ-fAV9OS9DESOZQtV/view?usp=sharing)

-[FLIR] [google drive](https://drive.google.com/file/d/1PwEOgT5ZOTjoKT2LpOzvCsxsVgwP8NIJ/view)


#### Change the data cfg
some example in data/multispectral/

#### Change the model cfg
some example in models/transformer/

note!!!   we used xxxx_transfomerx3_dataset.yaml in our paper.

## Results

|Dataset|CFT|mAP50|mAP75|mAP|
|:---------: |------------|:-----:|:-----------------:|:-------------:|
|FLIR||73.0|32.0|37.4|
|FLIR| ✔️ |**78.7 (Δ5.7)**|**35.5 (Δ3.5)**|**40.2 (Δ2.8)**|
|LLVIP||95.8|71.4|62.3|
|LLVIP| ✔️ |**97.5 (Δ1.7)**|**72.9 (Δ1.5)**|**63.6 (Δ1.3)**|
|VEDAI||79.7 | 47.7  | 46.8
|VEDAI| ✔️ |**85.3 (Δ5.6)**|**65.9(Δ18.2)**|**56.0 (Δ9.2)**|


### LLVIP
Log Average Miss Rate 
|Model| Log Average Miss Rate |
|:---------: |:--------------:|
|YOLOv3-RGB|37.70%|
|YOLOv3-IR|17.73%|
|YOLOv5-RGB|22.59%|
|YOLOv5-IR|10.66%|
|Baseline(Ours)|**6.91%**|
|CFT(Ours)|**5.40%**|

Miss Rate - FPPI curve
<div align="left">
<img src="https://github.com/DocF/multispectral-object-detection/blob/main/MR.png" width="500">
</div>

### Train Test and Detect
yolov3 yolov5 对应`train.py` `train_multi_modal.py` `test.py` `predict.py`脚本

yolov6 yolov8 yolov9 yolov11对应`train_v11.py` `train_multi_modal_v11.py` `test_v11.py` `predict_v11.py`脚本

其中models/yolo中的Model对应单模态模型，models/yolotest中的Model对应多模态模型

### 多模态的模型两种解决方案

第一种是add相加，如：`yolov5s_fusion_add.yaml` `PC2f_MPF_yolov8.yaml`

第二种是transformer CFT融合：`yolov5s_fusion_transformer.yaml`

### DDP多机多卡训练
在这个rep已经支持检测模型的多机多卡训练，使用torchrun的方式进行。如果需要检测nccl是否正常
可以使用`test_dist.py`这个脚本测试。

  
