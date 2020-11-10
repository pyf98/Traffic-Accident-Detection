# Traffic-Accident-Detection

This repository contains the python scripts written by Yifan Peng for the IDL course project.

## Introduction

In this project, traffic accident detection is interpreted as a binary classification task. Models can work on either individual frames or videos.

For frame-level classification, a pre-trained CNN is applied to every frame and the extracted features are classified by a multi-layer perceptron.

For video-level classification, a pre-trained CNN is used to extract spatial features and a multi-layer RNN is employed to capture temporal information.

## Usage

1. Install necessary packages.

2. Download videos from YouTube channels as described in the submitted report.

3. Modify the hyper-parameters in `TrainConfig.py`.

4. Start to train the CRNN model. The default model can run on a single GPU. Each epoch takes approximately 15 minutes on a NVIDIA Tesla T4 GPU.

>> python Train.py

5. Evaluate a trained model. First, set the path to the checkpoint in `Test.py`. Then, run the following

>> python Test.py

