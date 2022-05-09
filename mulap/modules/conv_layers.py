""" Code from https://github.com/minzwon/sota-music-tagging-models/ """

import torch.nn as nn


class Conv_1d(nn.Module):
    def __init__(self,
                 input_channels,
                 output_channels,
                 shape=3,
                 stride=1,
                 pooling=2):
        super(Conv_1d, self).__init__()
        self.conv = nn.Conv1d(input_channels,
                              output_channels,
                              shape,
                              stride=stride,
                              padding=shape // 2)
        self.bn = nn.BatchNorm1d(output_channels)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool1d(pooling)

    def forward(self, x):
        out = self.mp(self.relu(self.bn(self.conv(x))))
        return out


class Conv_V(nn.Module):
    # vertical convolution
    def __init__(self, input_channels, output_channels, filter_shape):
        super(Conv_V, self).__init__()
        self.conv = nn.Conv2d(input_channels,
                              output_channels,
                              filter_shape,
                              padding=(0, filter_shape[1] // 2))
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        freq = x.size(2)
        out = nn.MaxPool2d((freq, 1), stride=(freq, 1))(x)
        out = out.squeeze(2)
        return out


class Conv_H(nn.Module):
    # horizontal convolution
    def __init__(self, input_channels, output_channels, filter_length):
        super(Conv_H, self).__init__()
        self.conv = nn.Conv1d(input_channels,
                              output_channels,
                              filter_length,
                              padding=filter_length // 2)
        self.bn = nn.BatchNorm1d(output_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        freq = x.size(2)
        out = nn.AvgPool2d((freq, 1), stride=(freq, 1))(x)
        out = out.squeeze(2)
        out = self.relu(self.bn(self.conv(out)))
        return out
