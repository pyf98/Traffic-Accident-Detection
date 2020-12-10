# Traffic Accident Detection via Deep Learning


## ResNet101, VGG16_BN

Code in folder 'resnet101_frame': A fixed ResNet101 Conv2d CNN + 2 MLP layers

Code in folder 'vgg16bn_frame': A fixed VGG16_bn Conv2d CNN + 2 MLP layers

To run the training code:
```
python Train.py
```

To run the test code:
```
python Train.py
```

## Video-level ResNet101

Code in folder 'resnet101_video'

### Video-level Prediction
No training, just run test script:
```
python Train.py
```

### Visualization
This program generates and saves video-level plot of 4 video clips. Selected videos should be saved in folders 1, 2, 3, and 4 before running this script

```
python Testplot.py
```