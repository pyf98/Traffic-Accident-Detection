import torch
import torch.nn as nn
import torchvision.models as models


class FrameClassifier(nn.Module):
    '''2D CNN feature extractor based on pre-trained models.
    '''
    def __init__(self, cnn_type, fc_sizes, batchnorms, dropouts):
        super(FrameClassifier, self).__init__()

        assert cnn_type in ['resnet50', 'resnet101', 'resnet152'], f'invalid cnn type: {cnn_type}'
        # Note: for the following 3 types of ResNet, the output dim is 2048
        if cnn_type == 'resnet50':
            cnn = models.resnet50(pretrained=True)
        elif cnn_type == 'resnet101':
            cnn = models.resnet101(pretrained=True)
        else:
            cnn = models.resnet152(pretrained=True)

        modules = list(cnn.children())[:-1]         # remove the last FC layer
        self.cnn = nn.Sequential(*modules)
        in_features = cnn.fc.in_features            # 2048

        fc_layers = [nn.Dropout(dropouts[0])]       # input dropout
        dropouts = dropouts[1:]
        for hidden_size, batchnorm, drop_p in zip(fc_sizes, batchnorms, dropouts):
            fc_layers.append(nn.Linear(in_features, hidden_size))
            in_features = hidden_size
            if batchnorm:
                fc_layers.append(nn.BatchNorm1d(hidden_size))
            fc_layers.append(nn.LeakyReLU(inplace=True))
            fc_layers.append(nn.Dropout(p=drop_p))
        fc_layers.append(nn.Linear(in_features, 1))        # binary classification
        fc_layers.append(nn.Sigmoid())

        self.classifier = nn.Sequential(*fc_layers)

    def forward(self, x):
        with torch.no_grad():
            out = self.cnn(x)
            out = out.reshape(out.shape[0], -1)             # (N, 2048)

        prob = self.classifier(out)         # (N, 1)
        return prob
