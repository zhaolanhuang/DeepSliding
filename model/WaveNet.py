# [1] A. van den Oord et al., “WaveNet: A Generative Model for Raw Audio,” Sep. 19, 2016, arXiv: arXiv:1609.03499. doi: 10.48550/arXiv.1609.03499.
# [2] T. L. Paine et al., “Fast Wavenet Generation Algorithm,” Nov. 29, 2016, arXiv: arXiv:1611.09482. doi: 10.48550/arXiv.1611.09482.

# TODO
import torch
import torch.nn as nn

DEFAULT_INPUT_SHAPE = (64, 256) # (C, T)
DEFAULT_SLIDING_STEP_SIZE = 1

class WaveNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(64, 32, kernel_size=3, dilation=1)
    def forward(self, x):
        x = self.conv1(x)
        return x