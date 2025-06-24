# [1] M. Zanghieri, S. Benatti, A. Burrello, V. Kartsch, F. Conti, and L. Benini, “Robust Real-Time Embedded EMG Recognition Framework Using Temporal Convolutional Networks on a Multicore IoT Processor,” IEEE Trans. Biomed. Circuits Syst., vol. 14, no. 2, pp. 244–256, Apr. 2020, doi: 10.1109/TBCAS.2019.2959160.
# TODO

import torch
import torch.nn as nn

DEFAULT_INPUT_SHAPE = (1, 14 ,300) # (N, C, T), add N=1 for avoid tvm's error on Pool1d
DEFAULT_SLIDING_STEP_SIZE = 30

class TEMPONet_blocks(nn.Module):
    def __init__(self, dilation, stride, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels // 2, 3, dilation=dilation, padding=dilation)
        self.conv2 = nn.Conv1d(out_channels // 2, out_channels // 2, 3, dilation=dilation, padding=dilation)
        self.conv3 = nn.Conv1d(out_channels // 2, out_channels, 5, stride=stride, padding=2)
        self.avg_pool = nn.AvgPool1d(2, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.avg_pool(x)

class TEMPONet(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = TEMPONet_blocks(2, 1, 14, 32)
        self.block2 = TEMPONet_blocks(4, 2, 32, 64)
        self.block3 = TEMPONet_blocks(8, 4, 64, 128)
        self.flatten = nn.Flatten(start_dim=0, end_dim=-1)
        self.linear1 = nn.Linear(128*5,128)
        self.linear2 = nn.Linear(128, 8)
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return x
    

if __name__ == "__main__":
    x = torch.randn(*DEFAULT_INPUT_SHAPE)
    net = TEMPONet().eval()
    y = net(x)
    print(y.shape)