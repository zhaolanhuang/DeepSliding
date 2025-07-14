from typing import Any, Dict, Tuple, Type, Union, Optional

import torch
import torch.nn as nn
from .transformer_basic_blocks import TransformerModel, TransformerBlock
# from torch.nn.utils import weight_norm
DEFAULT_INPUT_SHAPE = (1, 1 ,48000) # (N, C, T), add N=1 for avoid tvm's error on Pool1d
DEFAULT_SLIDING_STEP_SIZE = 16000

class TinyChirpTransformerTime(nn.Module):
    def __init__(self, num_classes=2, n_embd=16, n_head=1, hidden_size=32, n_layers=1):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3)
        self.pool = nn.MaxPool1d(2, 2)
        #self.batch1 = nn.BatchNorm1d(16)
        self.dropout = nn.Dropout(0.25)
        self.relu = nn.ReLU()
        self.adpool = nn.AdaptiveAvgPool1d(1)

        self.blocks = nn.Sequential(*[TransformerBlock(n_embd, n_head, hidden_size) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, num_classes)
    

    def forward(self, x):
        #x = self.pool(self.relu(self.batch1(self.conv1(x))))
        
        x = self.pool(self.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.adpool(x)
        x = x.reshape(1, 1, -1)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits