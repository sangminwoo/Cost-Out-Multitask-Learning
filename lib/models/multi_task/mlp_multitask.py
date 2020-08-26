import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class MLP(nn.Module):
    def __init__(self, args, num_layers, input_size, hidden_size, output_size1=100, output_size2=1000, bn_momentum=1e-3, dropout=0.):
        super(MLP, self).__init__()
        self.layers = num_layers

        input_linear = nn.Linear(input_size, hidden_size)
        mlp_linear = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers-1)])
        
        if args.mode == 'filter':
            self.filter1 = nn.ParameterList([
                nn.ParameterDict({
                    'weight': nn.Parameter(torch.Tensor(1), requires_grad=True),
                    'bias': nn.Parameter(torch.Tensor(1), requires_grad=True)
                }) for _ in range(num_layers)
            ])
            self.filter2 = nn.ParameterList([
                nn.ParameterDict({
                    'weight': nn.Parameter(torch.Tensor(1), requires_grad=True),
                    'bias': nn.Parameter(torch.Tensor(1), requires_grad=True)
                }) for _ in range(num_layers)
            ])

            self.linear1 = nn.Linear(input_size, hidden_size)
            self.linear1.weight = input_linear.weight + self.filter1[0].weight
            self.linear1.bias = input_linear.bias + self.filter1[0].bias

            self.linear2 = nn.Linear(input_size, hidden_size)
            self.input2.weight = input_linear.weight + self.filter2[0].weight
            self.input2.bias = input_linear.bias + self.filter2[0].bias

            self.mlp_linear1 = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers-1)])
            for i in range(num_layers-1):
                self.mlp.linear1[i].weight = mlp_linear[i].weight + self.filter1[i+1].weight
                self.mlp.linear1[i].bias = mlp_linear[i].bias + self.filter1[i+1].weight

            self.mlp_linear2 = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers-1)])
            for i in range(num_layers-1):
                self.mlp.linear2[i].weight = mlp_linear[i].weight + self.filter2[i+1].weight
                self.mlp.linear2[i].bias = mlp_linear[i].bias + self.filter2[i+1].weight

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
        if self.layers > 2:
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

        self.out1 = nn.Linear(hidden_size, output_size1)
        self.out2 = nn.Linear(hidden_size, output_size2)
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
        if self.layers > 2:
            x1 = self.mlp1(x1)

        x2 = self.input2(x2)
        if self.layers > 2:
            x2 = self.mlp2(x2)

        out1 = self.out1(x1)
        out2 = self.out2(x2)
        # out = self.softmax(x)
        return out1, out2

def build_mlp(args, num_layers, input_size, hidden_size, output_size1, output_size2, bn_momentum, dropout):
    return MLP(args, num_layers, input_size, hidden_size, output_size1, output_size2, bn_momentum, dropout)


if __name__ == '__main__':
    x = torch.randn(3, 10)
    mlp = build_mlp(2, 10, 10, 10, 1e-3, 0.)
    out = mlp(x)
    print(out)