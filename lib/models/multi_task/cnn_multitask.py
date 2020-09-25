import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class CNN(nn.Module):
    def __init__(self, args, num_layers, input_channel, hidden_channel, output_size1=100, output_size2=1000, bn_momentum=1e-3, dropout=0.):
        super(CNN, self).__init__()
        self.layers = num_layers
        self.conv1 = nn.Conv2d(input_channel, hidden_channel, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(input_channel, hidden_channel, kernel_size=3, stride=1, padding=1)
        self.conv_layers1 = nn.ModuleList([
                nn.Conv2d(hidden_channel, hidden_channel, kernel_size=3, stride=1, padding=1) \
                for _ in range(num_layers-1)])
        self.conv_layers2 = nn.ModuleList([
                nn.Linear(hidden_channel, hidden_channel, kernel_size=3, stride=1, padding=1) \
                for _ in range(num_layers-1)])

        if args.mode == 'mt_filter':
            self.conv1 = nn.Linear(input_channel, hidden_channel)
            self.conv1.weight = nn.Parameter(self.conv1.weight + torch.Tensor(1), requires_grad=True)
            self.conv1.bias = nn.Parameter(self.conv1.bias + torch.Tensor(1), requires_grad=True)

            self.conv2 = nn.Linear(input_channel, hidden_channel)
            self.conv2.weight = nn.Parameter(self.conv2.weight + torch.Tensor(1), requires_grad=True)
            self.conv2.bias = nn.Parameter(self.conv2.bias + torch.Tensor(1), requires_grad=True)

            self.conv_layers1 = nn.ModuleList([nn.Linear(hidden_channel, hidden_channel) for _ in range(num_layers-1)])
            for i in range(num_layers-1):
                self.conv_layers1[i].weight = nn.Parameter(self.conv_layers1[i].weight + torch.Tensor(1), requires_grad=True)
                self.conv_layers1[i].bias = nn.Parameter(self.conv_layers1[i].bias + torch.Tensor(1), requires_grad=True)

            self.conv_layers2 = nn.ModuleList([nn.Linear(hidden_channel, hidden_channel) for _ in range(num_layers-1)])
            for i in range(num_layers-1):
                self.conv_layers2[i].weight = nn.Parameter(self.conv_layers2[i].weight + torch.Tensor(1), requires_grad=True)
                self.conv_layers2[i].bias = nn.Parameter(self.conv_layers2[i].bias + torch.Tensor(1), requires_grad=True)

        self.input1 = nn.Sequential(
                        self.conv1,
                        nn.BatchNorm2d(num_features=hidden_channel, momentum=bn_momentum),
                        nn.ReLU(True),
                        nn.Dropout(p=dropout),
                        nn.MaxPool2d(kernel_size=2, stride=2)
                    )
        self.input2 = nn.Sequential(
                        self.conv2,
                        nn.BatchNorm2d(num_features=hidden_channel, momentum=bn_momentum),
                        nn.ReLU(True),
                        nn.Dropout(p=dropout),
                        nn.MaxPool2d(kernel_size=2, stride=2)
                    )
        if self.layers >= 2:
            self.conv_block1 = nn.ModuleList([
                            nn.Sequential(
                                self.conv_layers1[i],
                                nn.BatchNorm2d(num_features=hidden_channel, momentum=bn_momentum),
                                nn.ReLU(True),
                                nn.Dropout(p=dropout),
                                nn.MaxPool2d(kernel_size=2, stride=2)
                            ) for i in range(num_layers-1)
                        ])
            self.conv_block2 = nn.ModuleList([
                            nn.Sequential(
                                self.conv_layers2[i],
                                nn.BatchNorm2d(num_features=hidden_channel, momentum=bn_momentum),
                                nn.ReLU(True),
                                nn.Dropout(p=dropout),
                                nn.MaxPool2d(kernel_size=2, stride=2)
                            ) for i in range(num_layers-1)
                        ])

        height = input_shape[1]
        width = input_shape[2]
        for _ in range(num_layers):
            height = height // 2
            width = width // 2

        self.output1 = nn.Linear(height*width*hidden_channel, output_size1)
        self.output2 = nn.Linear(height*width*hidden_channel, output_size2)
        self.softmax = nn.Softmax(dim=1)

        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         m.weight.data = nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('tanh'))
        #         m.bias.data = nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x1, x2):
        x1 = self.input1(x1)
        if self.layers >= 2:
            for conv_block1 in self.conv_block1:
                x1 = conv_block1(x1)

        x2 = self.input2(x2)
        if self.layers >= 2:
            for conv_block2 in self.conv_block2:
                x2 = conv_block2(x2)

        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        out1 = self.output1(x1)
        # out1 = self.softmax(out1)
        out2 = self.output2(x2)
        # out2 = self.softmax(out2)
        return out1, out2

def build_mt_cnn(args, num_layers, input_channel, hidden_channel, output_size1, output_size2, bn_momentum, dropout):
    return CNN(args, num_layers, input_channel, hidden_channel, output_size1, output_size2, bn_momentum, dropout)
