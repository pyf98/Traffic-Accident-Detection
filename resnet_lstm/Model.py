import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class CNNEncoder(nn.Module):
    '''2D CNN feature extractor based on pre-trained models.
    '''
    def __init__(self, dropout, embedding_dim, cnn_type):
        super(CNNEncoder, self).__init__()

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
        cnn_out_dim = cnn.fc.in_features        # 2048

        self.proj = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(cnn_out_dim, embedding_dim)
        )

    def forward(self, x_seq):
        '''
        :param x_seq: (N, T, C, H, W)
        '''

        feature_seq = []
        for t in range(x_seq.shape[1]):
            with torch.no_grad():                   # pre-trained model is fixed
                x = self.cnn(x_seq[:, t, :, :, :])
                x = x.reshape(x.shape[0], -1)       # (N, cnn_out_dim)

            x = self.proj(x)                        # (N, emb_dim)
            feature_seq.append(x)

        feature_seq = torch.stack(feature_seq, dim=0)   # (T, N, emb_dim)
        feature_seq = feature_seq.transpose(0, 1)       # (N, T, emb_dim)

        return feature_seq


class RNNDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, num_layers):
        super(RNNDecoder, self).__init__()

        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False
        )
        nn.init.xavier_normal_(self.rnn.all_weights[0][0])
        nn.init.xavier_normal_(self.rnn.all_weights[0][1])
        # For bidirectional RNNs
        # nn.init.xavier_normal_(self.rnn.all_weights[1][0])
        # nn.init.xavier_normal_(self.rnn.all_weights[1][1])

        # binary classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, feature_seq, seq_lens):
        '''
        :param feature_seq: (N, T, dim)
        :param seq_lens: (N,)
        '''

        self.rnn.flatten_parameters()       # for DataParallel

        packed_seq = pack_padded_sequence(feature_seq, seq_lens, batch_first=True, enforce_sorted=False)
        packed_out, (_, _) = self.rnn(packed_seq)
        out, out_lens = pad_packed_sequence(packed_out, batch_first=True)   # (N, T, hidden), (N,)
        last_out = []
        for b, l in enumerate(out_lens):
            last_out.append(out[b, l-1, :])
        last_out = torch.stack(last_out, dim=0)     # (N, hidden_size)

        # print(last_out[:5, :5])

        prob = self.classifier(last_out)        # (N, 1)
        return prob

    def predict_probs(self, feature_seq, seq_lens):
        '''Return probs at all time steps.
        :param feature_seq: (N, T, dim)
        :param seq_lens: (N,)
        :returns probs: (N, T), between 0 and 1
        :returns out_lens: (N,)
        '''
        self.rnn.flatten_parameters()  # for DataParallel

        packed_seq = pack_padded_sequence(feature_seq, seq_lens, batch_first=True, enforce_sorted=False)
        packed_out, (_, _) = self.rnn(packed_seq)
        out, out_lens = pad_packed_sequence(packed_out, batch_first=True)  # (N, T, hidden), (N,)
        # last_out = []
        # for b, l in enumerate(out_lens):
        #     last_out.append(out[b, l - 1, :])
        # last_out = torch.stack(last_out, dim=0)  # (N, hidden_size)
        #
        # # print(last_out[:5, :5])

        probs = self.classifier(out)            # (N, T, 1)
        probs = probs.squeeze(-1)               # (N, T)

        return probs, out_lens


class CRNNClassifier(nn.Module):
    def __init__(self, cnn_dropout, cnn_emb_dim, cnn_type, rnn_hidden_size, rnn_dropout, num_rnn_layers):
        super(CRNNClassifier, self).__init__()

        self.cnn_enc = CNNEncoder(cnn_dropout, cnn_emb_dim, cnn_type)
        self.rnn_dec = RNNDecoder(cnn_emb_dim, rnn_hidden_size, rnn_dropout, num_rnn_layers)

    def forward(self, x_seq, x_lens):
        '''
        :param x_seq: (N, T, C, H, W)
        :param x_lens: (N,)
        :return: prob of anomaly, (N, 1)
        '''

        feature_seq = self.cnn_enc(x_seq)       # (N, T, emb_dim)
        prob = self.rnn_dec(feature_seq, x_lens)

        return prob

    def predict_probs(self, x_seq, x_lens):
        '''
        :param x_seq: (N, T, C, H, W)
        :param x_lens: (N,)
        :returns probs: (N, T), between 0 and 1
        :returns out_lens: (N,)
        '''

        feature_seq = self.cnn_enc(x_seq)       # (N, T, emb_dim)
        probs, out_lens = self.rnn_dec.predict_probs(feature_seq, x_lens)

        return probs, out_lens
