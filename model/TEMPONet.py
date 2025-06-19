# [1] M. Zanghieri, S. Benatti, A. Burrello, V. Kartsch, F. Conti, and L. Benini, “Robust Real-Time Embedded EMG Recognition Framework Using Temporal Convolutional Networks on a Multicore IoT Processor,” IEEE Trans. Biomed. Circuits Syst., vol. 14, no. 2, pp. 244–256, Apr. 2020, doi: 10.1109/TBCAS.2019.2959160.
# TODO

import torch
import torch.nn as nn

DEFAULT_INPUT_SHAPE = (64, 256) # (C, T)
DEFAULT_SLIDING_STEP_SIZE = 1

class TEMPONet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(64, 32, kernel_size=3, dilation=1)
    def forward(self, x):
        x = self.conv1(x)
        return x