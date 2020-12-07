# Traffic-Accident-Detection

This repository contains the python scripts for the IDL course project.

TODO: The following will be updated in a few days.

## Introduction

In this project, traffic accident detection is interpreted as a binary classification task. Models can work on either individual frames or videos.

For frame-level classification, a pre-trained CNN is applied to every frame and the extracted features are classified by a multi-layer perceptron.

For video-level classification, a pre-trained CNN is used to extract spatial features and a multi-layer RNN is employed to capture temporal information.

## Requirements

```
numpy
tqdm
torchvision==0.7.0
torch==1.6.0
Pillow
scikit_learn
```

## Usage

### Frame-Level Classification: CNN + MLP

Scripts are in the folder `resnet_mlp`.

1. Install necessary packages.

2. Download videos from YouTube channels as described in the submitted report.

3. Modify the hyper-parameters in `TrainConfig.py`. For the CNN component, ResNet50, ResNet101, and ResNet152 are supported.

4. Start to train the CNN + MLP model. The default model can run on a single GPU. Each epoch takes approximately 30 minutes on a NVIDIA Tesla T4 GPU.

```
python Train.py
```

5. Evaluate a trained model. First, set the path to the checkpoint in `Test.py`. Then, run the following

```
python Test.py
```


### Video-Level Classification: CNN + RNN (CRNN)

Scripts are in the folder `resnet_lstm`.

1. Install necessary packages.

2. Download videos from YouTube channels as described in the submitted report.

3. Modify the hyper-parameters in `TrainConfig.py`. For the CNN component, ResNet50, ResNet101, and ResNet152 are supported.

4. Start to train the CRNN model. The default model can run on a single GPU. Each epoch takes approximately 15 minutes on a NVIDIA Tesla T4 GPU.

```
python Train.py
```

5. Evaluate a trained model. First, set the path to the checkpoint in `Test.py`. Then, run the following

```
python Test.py
```
