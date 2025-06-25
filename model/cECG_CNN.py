# [1] C. Hoog Antink, E. Breuer, D. Umutcan Uguz, and S. Leonhardt, “Signal-Level Fusion With Convolutional Neural Networks for Capacitively Coupled ECG in the Car,” presented at the 2018 Computing in Cardiology Conference, Dec. 2018. doi: 10.22489/CinC.2018.143.

import torch
import torch.nn as nn
# from torch.nn.utils import weight_norm
DEFAULT_INPUT_SHAPE = (1, 3 ,101) # (N, C, T), add N=1 for avoid tvm's error on Pool1d
DEFAULT_SLIDING_STEP_SIZE = 51

class cECGHeadBlock(nn.Module):
    def __init__(self):
        super(cECGHeadBlock, self).__init__()
        self.conv1 = nn.Conv1d(1, 10, 20)
        self.pool = nn.MaxPool1d(2, 2)
        self.conv2 = nn.Conv1d(10, 20, 15)
        self.conv3 = nn.Conv1d(20, 25, 10)
        self.conv4 = nn.Conv1d(25, 30, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x


class cECG_CNN(nn.Module):
    def __init__(self, input_channels=3):
        super(cECG_CNN, self).__init__()
        # self.input_channels=input_channels
        self.cnn_blocks = nn.ModuleList()
        self.flatten = nn.Flatten(start_dim=0, end_dim=-1)
        self.linear1 = nn.Linear(input_channels * 30 * 9, 100)
        self.linear2 = nn.Linear(100, 40)
        self.linear3 = nn.Linear(40, 2)
        for i in range(0, input_channels):
            self.cnn_blocks.append(cECGHeadBlock())

    def forward(self, x):
        block_outputs = []
        for i, cnn in enumerate(self.cnn_blocks):
            z = cnn(x[..., i:i+1, :])
            block_outputs.append(z)
        y = torch.stack(block_outputs)
        y = self.flatten(y)
        y = self.linear1(y)
        y = self.linear2(y)
        y = self.linear3(y)
        return y

if __name__ == "__main__":
    x = torch.randn(*DEFAULT_INPUT_SHAPE)
    net = cECG_CNN().eval()
    y = net(x)
    print(y.shape)