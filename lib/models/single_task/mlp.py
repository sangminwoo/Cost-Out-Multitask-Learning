import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class MLP(nn.Module):
    def __init__(self, args, num_layers, input_size, hidden_size, output_size, bn_momentum=1e-3, dropout=0.):
        super(MLP, self).__init__()
        self.layers = num_layers
        self.input = nn.Sequential(
                        nn.Linear(input_size, hidden_size),
                        nn.BatchNorm1d(num_features=hidden_size, momentum=bn_momentum),
                        nn.ReLU(True),
                        nn.Dropout(p=dropout)
                    )
        if self.layers >= 2:
            self.mlp = nn.ModuleList([
                            nn.Sequential(
                                nn.Linear(hidden_size, hidden_size),
                                nn.BatchNorm1d(num_features=hidden_size, momentum=bn_momentum),
                                nn.ReLU(True),
                                nn.Dropout(p=dropout)
                            ) for _ in range(num_layers-1)
                       ])

        self.output = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         m.weight.data = nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('tanh'))
        #         m.bias.data = nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)

        x = self.input(x)
        if self.layers >= 2:
            for mlp in self.mlp:
                x = mlp(x)

        out = self.output(x)
        # out = self.softmax(out)
        return out

def build_mlp(args, num_layers, input_size, hidden_size, output_size, bn_momentum, dropout):
    return MLP(args, num_layers, input_size, hidden_size, output_size, bn_momentum, dropout)

if __name__ == '__main__':
    x = torch.randn(3, 10)
    mlp = build_mlp(2, 10, 10, 10, 1e-3, 0.)
    out = mlp(x)
    print(out)