# [1] S. Bai, J. Z. Kolter, and V. Koltun, “An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling,” Apr. 19, 2018, arXiv: arXiv:1803.01271. doi: 10.48550/arXiv.1803.01271.
# taken from https://github.com/locuslab/TCN/blob/master/TCN/tcn.py

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
# Use Nottingham data shape by default
DEFAULT_INPUT_SHAPE = (1, 88 ,192) # (N, C, T), add N=1 for avoid tvm's error on Pool1d
DEFAULT_SLIDING_STEP_SIZE = 96

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, 
                                #  self.chomp1, 
                                 self.relu1, 
                                #  self.dropout1,
                                 self.conv2, 
                                #  self.chomp2, 
                                 self.relu2, 
                                #  self.dropout2
                                 )
        print(padding)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        print("out:", out.shape, "res:",res.shape)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size // 2, # ZL: slitly modify to align with Res conn
                                    # padding = 0,
                                     dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

INPUT_SIZE = 88
N_HIDDEN = 150
LEVELS = 4
N_CHANNELS = [N_HIDDEN] * LEVELS
KERNEL_SIZE = 5
DROPOUT = 0.25

# Poly Music model
class ResTCN(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, output_size=INPUT_SIZE, 
                 num_channels=N_CHANNELS, kernel_size=KERNEL_SIZE, dropout=DROPOUT):
        super(ResTCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # x needs to have dimension (N, C, L) in order to be passed into CNN
        output = self.tcn(x).transpose(1, 2)
        output = self.linear(output).double()
        return self.sig(output)
    
if __name__ == "__main__":
    x = torch.randn(*DEFAULT_INPUT_SHAPE)
    net = ResTCN().eval()
    y = net(x)
    print(y.shape)