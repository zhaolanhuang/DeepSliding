# [1] A. Burrello et al., “TCN Mapping Optimization for Ultra-Low Power Time-Series Edge Inference,” in 2021 IEEE/ACM International Symposium on Low Power Electronics and Design (ISLPED), Jul. 2021, pp. 1–6. doi: 10.1109/ISLPED52811.2021.9502494.
import torch
import torch.nn as nn



class ToyTCNd1(nn.Module):
    DEFAULT_INPUT_SHAPE = (64, 256) # (C, T)
    DEFAULT_SLIDING_STEP_SIZE = 1
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(64, 32, kernel_size=3, dilation=1)
    def forward(self, x):
        x = self.conv1(x)
        return x

class ToyTCNd2(nn.Module):
    DEFAULT_INPUT_SHAPE = (64, 256) # (C, T)
    DEFAULT_SLIDING_STEP_SIZE = 1
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(64, 32, kernel_size=3, dilation=2)
    def forward(self, x):
        x = self.conv1(x)
        return x