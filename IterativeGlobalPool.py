import torch
import torch.nn as nn
from .utils import ceildiv
from .utils import WaitForNextInputError

import torch
from torch.library import Library, impl

# pool_type: "Avg" or "Max"
# num_of_latent_state = ceil(pool_size / stride)
# stride is equals to deep sliding window size
class IterativeGlobalPool(nn.Module):
    def __init__(self, pool_type, pool_size, latent_dim, stride, bf16=False):
        super().__init__()
        self.buffer_size = ceildiv(pool_size, stride)
        self.stride = stride
        self.latent_dim = latent_dim
        self.pool_size = pool_size
        self.pool_type = pool_type
        
        self._num_cur_pool_filled = 0
        self._num_cur_cell_filled = 0
        self._cur_cell_idx = 0

        if bf16:
            self.buffer = torch.zeros(( [*self.latent_dim, self.buffer_size]), dtype=torch.bfloat16)
        else:
            self.buffer = torch.zeros(( [*self.latent_dim, self.buffer_size]), dtype=torch.float32)

        self.cell_pool_func = None
        self.output_func = None

        if self.pool_type == "Avg":
            self.cell_pool_func = lambda a, b: a + (b / pool_size)
            self.output_func = lambda a: torch.sum(a, -1, keepdim=True)
        elif self.pool_type == "Max":
            self.cell_pool_func = lambda a, b: torch.maximum(a, b)
            self.output_func = lambda a: torch.max(a, -1, keepdim=True)
        else:
            raise NotImplementedError(f"IterativeGlobalPool type: {self.pool_type} is not supported.")
    
    def forward(self, x):
        self.buffer[..., self._cur_cell_idx] = self.cell_pool_func(self.buffer[..., self._cur_cell_idx], x[..., -1])
        
        self._num_cur_cell_filled += 1
        if self._num_cur_cell_filled >= self.stride:
            self._num_cur_cell_filled = 0
            self._cur_cell_idx += 1
            self._cur_cell_idx %= self.buffer_size
        
        self._num_cur_pool_filled += 1
        if self._num_cur_pool_filled >= self.pool_size:
            self._num_cur_pool_filled -= self.stride
            x = self.output_func(self.buffer)
            return x
        else:
            raise WaitForNextInputError()

from .utils import torch_lib
# Create a library
my_lib = torch_lib

# Register an operator
my_lib.define("iterative_global_pool_op(Tensor x, str pool_type,int pool_size, int[] latent_dim, int stride) -> Tensor")

# Register CPU implementation
@impl("DeepSliding::iterative_global_pool_op", "CPU")
def iterative_global_pool_op_impl(x: torch.Tensor, pool_type: str, pool_size: int, latent_dim: list[int], stride: int) -> torch.Tensor:
    target_shape = list(latent_dim) + [1]
    return x.new_zeros(target_shape)


# Access operator handle
iterative_global_pool_op = torch.ops.DeepSliding.iterative_global_pool_op


class IterativeGlobalPoolFake(nn.Module):
    def __init__(self, pool_type, pool_size, latent_dim, stride, bf16=False):
        super().__init__()
        self.stride = stride
        self.latent_dim = latent_dim
        self.pool_size = pool_size
        self.pool_type = pool_type
    
    def forward(self, x):
        return iterative_global_pool_op(x, self.pool_type, self.pool_size, self.latent_dim, self.stride)


    
