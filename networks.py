import math
import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo


#######################################
#      CRNN Model for RGB Frames      #
#######################################

class CNNEncoder(nn.Module):
    '''2D CNN feature extractor based on pre-trained models.
    '''
    def __init__(self, dropout, embedding_dim, cnn_type, finetune):
        super(CNNEncoder, self).__init__()

        self.finetune = finetune        # if True, the CNN will also be trained

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
        cnn_out_dim = cnn.fc.in_features            # 2048

        self.proj = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(cnn_out_dim, embedding_dim)
        )

    def forward(self, x_seq):
        '''
        :param x_seq: (N, T, C, H, W)
        :return (N, T, emb_dim)
        '''

        feature_seq = []
        for t in range(x_seq.shape[1]):
            if self.finetune:
                x = self.cnn(x_seq[:, t, :, :, :])
                x = x.reshape(x.shape[0], -1)  # (N, cnn_out_dim)
            else:
                with torch.no_grad():                   # pre-trained model is fixed
                    x = self.cnn(x_seq[:, t, :, :, :])
                    x = x.reshape(x.shape[0], -1)       # (N, cnn_out_dim)

            x = self.proj(x)                        # (N, emb_dim)
            feature_seq.append(x)

        feature_seq = torch.stack(feature_seq, dim=0)   # (T, N, emb_dim)
        feature_seq = feature_seq.transpose(0, 1)       # (N, T, emb_dim)

        return feature_seq


class RNNDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, num_layers, bidirectional):
        super(RNNDecoder, self).__init__()

        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )
        nn.init.xavier_normal_(self.rnn.all_weights[0][0])
        nn.init.xavier_normal_(self.rnn.all_weights[0][1])
        # For bidirectional RNNs
        # nn.init.xavier_normal_(self.rnn.all_weights[1][0])
        # nn.init.xavier_normal_(self.rnn.all_weights[1][1])

        # binary classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size*2 if bidirectional else hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, feature_seq):
        '''
        :param feature_seq: (N, T, dim)
        :return out: (N, T), probability after sigmoid
        '''

        self.rnn.flatten_parameters()       # for DataParallel

        out, _ = self.rnn(feature_seq)                  # (N, T, hidden_size)
        out = self.classifier(out).squeeze(-1)          # (N, T), after sigmoid

        return out


class CRNNClassifier(nn.Module):
    def __init__(self, cnn_dropout, cnn_emb_dim, cnn_type, cnn_finetune,
                 rnn_hidden_size, rnn_dropout, num_rnn_layers, rnn_bidir):
        super(CRNNClassifier, self).__init__()

        self.cnn_enc = CNNEncoder(cnn_dropout, cnn_emb_dim, cnn_type, cnn_finetune)
        self.rnn_dec = RNNDecoder(cnn_emb_dim, rnn_hidden_size, rnn_dropout, num_rnn_layers, rnn_bidir)

    def forward(self, x_seq):
        '''
        :param x_seq: (N, T, C, H, W)
        :return: prob of anomaly after sigmoid, (N, T)
        '''

        feature_seq = self.cnn_enc(x_seq)       # (N, T, emb_dim)
        prob = self.rnn_dec(feature_seq)        # (N, T), probability after sigmoid

        return prob


##################################
#      CNN for Optical FLow      #
##################################
"""This part is similar to the temporal/motion stream in two-stream methods.
Here, only three types of ResNet are supported.

Ref:
    https://pytorch.org/docs/stable/torchvision/models.html?highlight=resnet
    https://github.com/jeffreyyihuang/two-stream-action-recognition
"""

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'
}

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, nb_classes=1, channel=20):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1_custom = nn.Conv2d(channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc_custom = nn.Linear(512 * block.expansion, nb_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1_custom(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        out = self.fc_custom(x)
        return out

    def extract_feature_vector(self, x):
        """Extract a feature vector from the input image.
        Args:
            x (torch.Tensor): (N, C, H, W)
        Returns:
            out (torch.Tensor): (N, 2048)
        """
        x = self.conv1_custom(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        out = x.view(x.size(0), -1)
        return out


def resnet50(pretrained=True, channel=20):
    model = ResNet(Bottleneck, [3, 4, 6, 3], nb_classes=1, channel=channel)
    if pretrained:
        pretrain_dict = model_zoo.load_url(model_urls['resnet50'])                  # modify pretrain code
        model_dict = model.state_dict()
        model_dict = weight_transform(model_dict, pretrain_dict, channel)
        model.load_state_dict(model_dict)
    return model


def resnet101(pretrained=True, channel=20):
    model = ResNet(Bottleneck, [3, 4, 23, 3], nb_classes=1, channel=channel)
    if pretrained:
        pretrain_dict = model_zoo.load_url(model_urls['resnet101'])                  # modify pretrain code
        model_dict = model.state_dict()
        model_dict = weight_transform(model_dict, pretrain_dict, channel)
        model.load_state_dict(model_dict)
    return model


def resnet152(pretrained=True, channel=20):
    model = ResNet(Bottleneck, [3, 8, 36, 3], nb_classes=1, channel=channel)
    if pretrained:
        pretrain_dict = model_zoo.load_url(model_urls['resnet152'])                  # modify pretrain code
        model_dict = model.state_dict()
        model_dict = weight_transform(model_dict, pretrain_dict, channel)
        model.load_state_dict(model_dict)
    return model


def cross_modality_pretrain(conv1_weight, channel):
    """Transforms the original 3 channel weight to "channel" channels
    """
    S=0
    for i in range(3):
        S += conv1_weight[:, i, :, :]
    avg = S / 3.
    new_conv1_weight = torch.FloatTensor(64, channel, 7, 7)
    for i in range(channel):
        new_conv1_weight[:, i, :, :] = avg.data
    return new_conv1_weight


def weight_transform(model_dict, pretrain_dict, channel):
    weight_dict  = {k:v for k, v in pretrain_dict.items() if k in model_dict}
    w3 = pretrain_dict['conv1.weight']
    if channel == 3:
        wt = w3
    else:
        wt = cross_modality_pretrain(w3, channel)

    weight_dict['conv1_custom.weight'] = wt
    model_dict.update(weight_dict)
    return model_dict


###################################
#      CRNN for Optical Flow      #
###################################

class CRNNOpticalFlow(nn.Module):
    def __init__(self, cnn_dropout, cnn_emb_dim, cnn_type,
                 rnn_hidden_size, rnn_dropout, num_rnn_layers, rnn_bidir):
        super(CRNNOpticalFlow, self).__init__()

        assert cnn_type in ['resnet50', 'resnet101', 'resnet152']
        if cnn_type == 'resnet50':
            self.cnn = resnet50(pretrained=True, channel=2)     # only 2 channels, i.e. x and y
        elif cnn_type == 'resnet101':
            self.cnn = resnet101(pretrained=True, channel=2)
        elif cnn_type == 'resnet152':
            self.cnn = resnet152(pretrained=True, channel=2)

        self.cnn.fc_custom = None       # here, we don't need the last linear layer

        self.embed = nn.Sequential(
            nn.Dropout(p=cnn_dropout),
            nn.Linear(2048, cnn_emb_dim),
            nn.Dropout(p=rnn_dropout)
        )

        self.rnn = RNNDecoder(cnn_emb_dim, rnn_hidden_size, rnn_dropout, num_rnn_layers, rnn_bidir)

    def forward(self, x):
        '''
        Args:
            x (torch.Tensor): (N, T * 2, H, W)
        Returns:
            prob: probability of anomaly after sigmoid, (N, T)
        '''

        N, _, H, W = x.shape
        x = x.reshape(N, -1, 2, H, W).reshape(-1, 2, H, W)
        x = self.cnn.extract_feature_vector(x)      # (N * T, 2048)
        x = x.reshape(N, -1, 2048)

        x = self.embed(x)                           # (N, T, emb_dim)

        prob = self.rnn(x)                          # (N, T), probability after sigmoid
        return prob


if __name__ == '__main__':
    model = resnet101(pretrained=True, channel=20)
    print(model)
