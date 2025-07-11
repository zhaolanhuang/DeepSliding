import torch
import torch.nn as nn


class BaseSSMOperator(nn.Module):
    def __init__(self, wrapped_operator: nn.Module, num_of_latent_state, latent_dim, stride):
        super().__init__()
        self._wrapped_operator = wrapped_operator
        self._num_of_latent_state = num_of_latent_state
        self._stride = stride
        self._latent_dim = latent_dim
    
    @property
    def stride(self):
        return self._stride
    
    def set_stride(self, s: int):
        if s > self._num_of_latent_state:
            raise RuntimeError("SSM: try to set stride larger than num of latent_state!")
        self._stride = s

    @property
    def num_of_latent_state(self):
        return self._num_of_latent_state