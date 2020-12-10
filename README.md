# Traffic-Accident-Detection

This repository contains the scripts for our IDL course project: **Traffic Accident Detection via Deep Learning**.


## Introduction

Detecting anomalous events such as road accidents in natural driving scenes is a challenging task. The majority of previous studies focus on fixed cameras with static backgrounds. In this project, we design **a CRNN-based two-stream method using both RGB frames and optical flow to detect traffic accidents in first-person dash-cam videos**. Our hypotheses are that motion features can improve the detection performance and that CRNN-based approaches are better for modeling temporal relationship than conventional CNN-based approaches. Results show that the motion stream outperforms the spatial-temporal stream, and that the fusion of two streams can further improve our model's performance.

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

## Models


