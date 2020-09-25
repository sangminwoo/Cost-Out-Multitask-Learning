import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class MLP(nn.Module):
    def __init__(self, args, num_layers, input_size, hidden_size, output_size1=100, output_size2=1000, bn_momentum=1e-3, dropout=0.):
        super(MLP, self).__init__()
        self.layers = num_layers
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(input_size, hidden_size)
        self.mlp_linear1 = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers-1)])
        self.mlp_linear2 = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers-1)])

        if args.mode == 'mt_filter':
            self.linear1 = nn.Linear(input_size, hidden_size)
            self.linear1.weight = nn.Parameter(self.linear1.weight + torch.Tensor(1), requires_grad=True)
            self.linear1.bias = nn.Parameter(self.linear1.bias + torch.Tensor(1), requires_grad=True)

            self.linear2 = nn.Linear(input_size, hidden_size)
            self.linear2.weight = nn.Parameter(self.linear2.weight + torch.Tensor(1), requires_grad=True)
            self.linear2.bias = nn.Parameter(self.linear2.bias + torch.Tensor(1), requires_grad=True)

            self.mlp_linear1 = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers-1)])
            for i in range(num_layers-1):
                self.mlp_linear1[i].weight = nn.Parameter(self.mlp_linear1[i].weight + torch.Tensor(1), requires_grad=True)
                self.mlp_linear1[i].bias = nn.Parameter(self.mlp_linear1[i].bias + torch.Tensor(1), requires_grad=True)

            self.mlp_linear2 = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers-1)])
            for i in range(num_layers-1):
                self.mlp_linear2[i].weight = nn.Parameter(self.mlp_linear2[i].weight + torch.Tensor(1), requires_grad=True)
                self.mlp_linear2[i].bias = nn.Parameter(self.mlp_linear2[i].bias + torch.Tensor(1), requires_grad=True)

        self.input1 = nn.Sequential(
                        self.linear1,
                        nn.BatchNorm1d(num_features=hidden_size, momentum=bn_momentum),
                        nn.ReLU(True),
                        nn.Dropout(p=dropout)
                    )
        self.input2 = nn.Sequential(
                        self.linear2,
                        nn.BatchNorm1d(num_features=hidden_size, momentum=bn_momentum),
                        nn.ReLU(True),
                        nn.Dropout(p=dropout)
                    )
        if self.layers >= 2:
            self.mlp1 = nn.ModuleList([
                            nn.Sequential(
                                self.mlp_linear1[i],
                                nn.BatchNorm1d(num_features=hidden_size, momentum=bn_momentum),
                                nn.ReLU(True),
                                nn.Dropout(p=dropout)
                            ) for i in range(num_layers-1)
                        ])
            self.mlp2 = nn.ModuleList([
                            nn.Sequential(
                                self.mlp_linear2[i],
                                nn.BatchNorm1d(num_features=hidden_size, momentum=bn_momentum),
                                nn.ReLU(True),
                                nn.Dropout(p=dropout)
                            ) for i in range(num_layers-1)
                        ])

        self.output1 = nn.Linear(hidden_size, output_size1)
        self.output2 = nn.Linear(hidden_size, output_size2)
        # self.softmax = nn.Softmax(dim=1)

        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         m.weight.data = nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('tanh'))
        #         m.bias.data = nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x1, x2):
        if len(x1.shape) > 2:
            x1 = x1.view(x1.size(0), -1)
        if len(x2.shape) > 2:
            x2 = x2.view(x2.size(0), -1)

        x1 = self.input1(x1)
        if self.layers >= 2:
            for mlp1 in self.mlp1:
                x1 = mlp1(x1)

        x2 = self.input2(x2)
        if self.layers >= 2:
            for mlp2 in self.mlp2:
                x2 = mlp2(x2)

        out1 = self.output1(x1)
        # out1 = self.softmax(out1)
        out2 = self.output2(x2)
        # out2 = self.softmax(out2)
        return out1, out2

def build_mt_mlp(args, num_layers, input_size, hidden_size, output_size1, output_size2, bn_momentum, dropout):
    return MLP(args, num_layers, input_size, hidden_size, output_size1, output_size2, bn_momentum, dropout)