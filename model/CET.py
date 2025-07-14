# [1] M. Rohr et al., “Transformer Network with Time Prior for Predicting Clinical Outcome from EEG of Cardiac Arrest Patients,” presented at the 2023 Computing in Cardiology Conference, Nov. 2023. doi: 10.22489/CinC.2023.173.

from typing import Any, Dict, Tuple, Type, Union, Optional

import torch
import torch.nn as nn
from .transformer_basic_blocks import TransformerModel
# from torch.nn.utils import weight_norm
DEFAULT_INPUT_SHAPE = (1, 18 ,320) # (N, C, T), add N=1 for avoid tvm's error on Pool1d
DEFAULT_SLIDING_STEP_SIZE = 51

#taken from https://github.com/ChristophReich1996/ECG_Classification/blob/main/ecg_classification/model.py
class Conv1dResidualBlock(nn.Module):
    """
    This class implements a simple residal block with 1d convolutions.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride=1, padding: int = 1,
                 bias: bool = False, convolution: Type[nn.Module] = nn.Conv1d,
                 normalization: Type[nn.Module] = nn.BatchNorm1d, activation: Type[nn.Module] = nn.LeakyReLU,
                 pooling: Tuple[nn.Module] = nn.AvgPool1d, dropout: float = 0.0) -> None:
        """
        Constructor method
        :param in_channels: (int) Number of input channels
        :param out_channels: (int) Number of output channels
        :param kernel_size: (int) Kernel size to be used in convolution
        :param stride: (int) Stride factor to be used in convolution
        :param padding: (int) Padding to be used in convolution
        :param bias: (int) If true bias is utilized in each convolution
        :param convolution: (Type[nn.Conv1d]) Type of convolution to be utilized
        :param normalization: (Type[nn.Module]) Type of normalization to be utilized
        :param activation: (Type[nn.Module]) Type of activation to be utilized
        :param pooling: (Type[nn.Module]) Type of pooling layer to be utilized
        :param dropout: (float) Dropout rate to be applied
        """
        # Call super constructor
        super(Conv1dResidualBlock, self).__init__()
        # Init main mapping
        self.main_mapping = nn.Sequential(
            convolution(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride,
                        padding=padding, bias=bias),
            normalization(num_features=in_channels, track_running_stats=True, affine=True), #ZL: keep as much channel at first
            activation(),
            nn.Dropout(p=dropout),
            convolution(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                        padding=padding, bias=bias),
            normalization(num_features=out_channels, track_running_stats=True, affine=True),
        )
        # Init residual mapping
        self.residual_mapping = convolution(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1,
                                            padding=0, bias=False) if in_channels != out_channels else nn.Identity()
        # Init final activation
        self.final_activation = activation()
        # Init final dropout
        self.dropout = nn.Dropout(p=dropout)
        # Init downsampling layer
        self.pooling = pooling(kernel_size=2, stride=2) if pooling != None else nn.Identity()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param input: (torch.Tensor) Input tensor [batch size, in channels, height]
        :return: (torch.Tensor) Output tensor
        """
        # Perform main mapping
        output = self.main_mapping(input)
        # Perform skip connection
        output = output + self.residual_mapping(input)
        # Perform final activation
        output = self.final_activation(output)
        # Perform final dropout
        output = self.dropout(output)
        # Perform final downsampling
        return self.pooling(output)

class CET_S(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = Conv1dResidualBlock(18, 1, 5, 1, 2)
        self.proj = nn.Linear(160, 64)
        self.transformer = TransformerModel(num_classes=64, n_embd=64, n_head=1, hidden_size=64, n_layers=4)
        self.fc1 = nn.Linear(64, 2)
        self.flatten = nn.Flatten(end_dim=-1) # TODO should work without flatten in future
    def forward(self, x):
        x = self.resnet(x)
        x = self.flatten(x)
        x: torch.Tensor = self.proj(x)
        x = x.reshape(1,1,-1) # Love from TVM: "Only 3D or 4D query supported"
        x = self.transformer(x)
        return self.fc1(x)

class CET(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = Conv1dResidualBlock(18, 1, 5, 1, 2)
        self.proj = nn.Linear(160, 256)
        self.transformer = TransformerModel(num_classes=256, n_embd=256, n_head=1, hidden_size=64, n_layers=4)
        self.fc1 = nn.Linear(256, 2)
    def forward(self, x):
        x = self.resnet(x)
        x = self.proj(x)
        x = self.transformer(x)
        return self.fc1(x)
    
class CET_VAR(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = Conv1dResidualBlock(18, 1, 5, 1, 2)
        self.transformer = TransformerModel(num_classes=320, n_embd=320, n_head=1, hidden_size=64, n_layers=4)
        self.fc1 = nn.Linear(320, 2)
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1) # TODO should work without flatten in future
    def forward(self, x):
        x = self.resnet(x)
        x = self.flatten(x)
        x = self.transformer(x)
        return self.fc1(x)

if __name__ == "__main__":
    x = torch.randn(*DEFAULT_INPUT_SHAPE)
    net = CET().eval()
    y = net(x)
    print(y.shape)