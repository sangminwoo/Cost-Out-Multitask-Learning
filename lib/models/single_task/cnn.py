import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class CNN(nn.Module):
    def __init__(self, args, num_layers, input_channel, hidden_channel, output_channel1=100, output_channel2=1000, bn_momentum=1e-3, dropout=0.):
        super(CNN, self).__init__()
        self.layers = num_layers
        self.input = nn.Sequential(
                        nn.Conv2d(input_channel, hidden_channel, kernel_channel=3, stride=1, padding=1),
                        nn.BatchNorm2d(num_features=hidden_channel, momentum=bn_momentum),
                        nn.ReLU(True),
                        nn.Dropout(p=dropout),
                        nn.MaxPool2d(kernel_size=2, stride=1)
                    )
        if self.layers >= 2:
            self.cnn = nn.ModuleList([
                            nn.Sequential(
                                nn.Conv2d(input_channel, hidden_channel, kernel_channel=3, stride=1, padding=1),
                                nn.BatchNorm2d(num_features=hidden_channel, momentum=bn_momentum),
                                nn.ReLU(True),
                                nn.Dropout(p=dropout),
                                nn.MaxPool2d(kernel_size=2, stride=1)
                            ) for _ in range(num_layers-1)
                       ])

        self.output = nn.Linear(hidden_channel, output_channel)
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
        x = self.output(x)
        # out = self.softmax(x)
        return out

def build_cnn(args, num_layers, input_channel, hidden_channel, output_channel, bn_momentum, dropout):
    return CNN(args, num_layers, input_channel, hidden_channel, output_channel, bn_momentum, dropout)