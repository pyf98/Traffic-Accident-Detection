# Traffic Accident Detection via Deep Learning

This repository contains the code of our IDL course project in Fall 2020.


## Introduction

Detecting anomalous events such as road accidents in natural driving scenes is a challenging task. The majority of previous studies focus on fixed cameras with static backgrounds. In this project, we design **a CRNN-based two-stream method using both RGB frames and optical flow to detect traffic accidents in first-person dash-cam videos**. Our hypotheses are that motion features can improve the detection performance and that CRNN-based approaches are better for modeling temporal relationship than conventional CNN-based approaches. Results show that the motion stream outperforms the spatial-temporal stream, and that the fusion of two streams can further improve our model's performance.

![two-stream](imgs/crnn_twostream.png "CRNN-based two-stream method for traffic accident detection")

## Requirements

Our models are implemented using PyTorch. Required packages are listed in `requirements.txt`.

```
numpy
tqdm
torchvision==0.7.0
torch==1.6.0
Pillow
scikit_learn
```

To install these packages, run

```
pip install -r requirements.txt
```

All models can be trained on a single NVIDIA Tesla T4 GPU using the default configuration.

## Dataset

We employ a recently introduced traffic anomaly dataset called [Detection of Traffic Anomaly](https://github.com/MoonBlvd/Detection-of-Traffic-Anomaly "Detection of Traffic Anomaly Dataset") (DoTA). DoTA contains 4,677 dash-cam videos collected from YouTube channels. These ego-centric driving videos are from different countries and under different weather and lighting conditions.

**Note that due to issues with YouTube, some videos are no longer available. We have collected 4,085 videos in total.** Most videos in DoTA can be separated into three temporal partitions: precursor, anomaly window, and post-anomaly. We label the first part (i.e. precursor) as *normal* or *non-accident*, and the second part (i.e. anomaly window) as *anomaly* or *accident*, but we do not use the third part. Details of our dataset are shown below.

Dataset | Training | Validation | Test
:---: | :---: | :---: | :---:
\#video clips | 5,700 | 801 | 1,657
\#frames | 208,649 | 29,997 | 58,778


## Models

### Spatial-Temporal Stream

The spatial-temporal stream takes RGB frames as input, which contain appearance information. To extract frame-level features from an input video, an ImageNet pre-trained ResNet is applied. To capture high-level (temporal) information, three architectures are employed: a multi-layer perceptron (MLP), a unidirectional Long Short-Term Memory (LSTM), and a bidirectional LSTM (BiLSTM). The MLP doesn't consider temporal dependencies, which leads to degraded performance.

* ResNet + MLP: 

* ResNet + LSTM:

* ResNet + BiLSTM:


### Motion Stream


### Fusion of Two Streams



