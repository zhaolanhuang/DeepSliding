import torch
import torch.nn as nn
from .CircularBuffer import CircularBuffer
from .BaseSSMOperator import BaseSSMOperator
from .utils import WaitForNextInputError

class SSMOperator(BaseSSMOperator):
    def __init__(self, wrapped_operator: nn.Module, num_of_latent_state, latent_dim, stride):
        super(SSMOperator, self).__init__(wrapped_operator, num_of_latent_state, latent_dim, stride)
        self._cir_buffer = CircularBuffer(num_of_latent_state, latent_dim)
        self._current_filled_latent_state = 0
    # 1D: (C, T) scheme, 2D: (C,H,W,T) scheme -> (Latent_DIM, T) scheme, T for time axis
    # Input: (C, 1) or (C, H, W, 1) -> point-by-point along time
    # Return: (..., 1) for Operator output on the latent state, when buffer of latent state is ready (filled).
    # Exception: WaitForNextInputError -> latent state is not filled, wait for the next input.
    def forward(self, x):
        x = self._cir_buffer(x)
        self._current_filled_latent_state += 1
        if self._current_filled_latent_state >= self._num_of_latent_state:
            # print("cir buf:", x)
            x = self._wrapped_operator(x)
            # print(self._wrapped_operator, x)
            self._current_filled_latent_state -= self._stride
        else:
            raise WaitForNextInputError()
        return x