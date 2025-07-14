import torch
import torch.nn as nn

import torch
from torch.library import Library, impl

import copy

from .BaseSSMOperator import BaseSSMOperator

# Create a library
my_lib = Library("DeepSliding", "DEF")

# Register an operator
my_lib.define("ssm_fake_op(Tensor x, int num_of_latent_state, int[] latent_dim, int stride) -> Tensor")

# Register CPU implementation
@impl("DeepSliding::ssm_fake_op", "CPU")
def ssm_fake_op_impl(x: torch.Tensor, num_of_latent_state: int, latent_dim: list[int], stride: int) -> torch.Tensor:
    target_shape = list(latent_dim) + [num_of_latent_state]
    return x.new_zeros(target_shape)


# Access operator handle
ssm_fake_op = torch.ops.DeepSliding.ssm_fake_op


class SSMFakeOperator(BaseSSMOperator):
    def __init__(self, wrapped_operator: nn.Module, num_of_latent_state, latent_dim, stride):
        super(SSMFakeOperator, self).__init__(wrapped_operator, num_of_latent_state, latent_dim, stride)
        self._wrapped_operator = copy.deepcopy(wrapped_operator) 
        # TODO: should be more elegant to do so..
        if isinstance(self._wrapped_operator, nn.Conv1d):
            self._wrapped_operator.padding = 0

    # 1D: (C, T) scheme, 2D: (C,H,W,T) scheme -> (Latent_DIM, T) scheme, T for time axis
    # Input: (C, 1) or (C, H, W, 1) -> point-by-point along time
    # Return: (..., 1) for Operator output on the latent state, when buffer of latent state is ready (filled).
    # Exception: WaitForNextInputError -> latent state is not filled, wait for the next input.
    def forward(self, x):
        x = ssm_fake_op(x, self._num_of_latent_state, self._latent_dim, self._stride)
        return self._wrapped_operator(x)