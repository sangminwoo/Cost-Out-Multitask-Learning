import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class CNN(nn.Module):
    def __init__(self, args, num_layers, input_shape, channel_size, output_size=100, bn_momentum=1e-3, dropout=0.):
        super(CNN, self).__init__()
        self.layers = num_layers
        self.input = nn.Sequential(
                        nn.Conv2d(input_shape[0], channel_size, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(num_features=channel_size, momentum=bn_momentum),
                        nn.ReLU(True),
                        nn.Dropout(p=dropout),
                        nn.MaxPool2d(kernel_size=2, stride=2)
                    )
        if self.layers >= 2:
            self.cnn = nn.ModuleList([
                            nn.Sequential(
                                nn.Conv2d(channel_size, channel_size, kernel_size=3, stride=1, padding=1),
                                nn.BatchNorm2d(num_features=channel_size, momentum=bn_momentum),
                                nn.ReLU(True),
                                nn.Dropout(p=dropout),
                                nn.MaxPool2d(kernel_size=2, stride=2)
                            ) for _ in range(num_layers-1)
                       ])

        height = input_shape[1]
        width = input_shape[2]
        for _ in range(num_layers):
            height = height // 2
            width = width // 2

        self.output = nn.Linear(height*width*channel_size, output_size)
        self.softmax = nn.Softmax(dim=1)

        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         m.weight.data = nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('tanh'))
        #         m.bias.data = nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        x = self.input(x)
        if self.layers >= 2:
            for cnn in self.cnn:
                x = cnn(x)

        x = x.view(x.size(0), -1)
        out = self.output(x)
        # out = self.softmax(out)
        return out

def build_cnn(args, num_layers, input_shape, channel_size, output_size, bn_momentum, dropout):
    return CNN(args, num_layers, input_shape, channel_size, output_size, bn_momentum, dropout)