# Traffic-Accident-Detection

This repository contains the python scripts written by Yifan Peng for the IDL course project.

## Introduction

In this project, traffic accident detection is interpreted as a binary classification task. Models can work on either individual frames or videos. For frame-level classification, a pre-trained CNN is applied to every frame and the extracted features are classified by a multi-layer perceptron. For video-level classification, a pre-trained CNN is used to extract spatial features and a multi-layer RNN is employed to capture temporal information.

## Usage

