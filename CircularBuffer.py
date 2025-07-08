import torch
import torch.nn as nn


class CircularBuffer(nn.Module):
    def __init__(self, buffer_size: int, latent_dim=None, bf16=True):
        super().__init__()
        self.buffer_size = buffer_size
        self.latent_dim = latent_dim
        # print(buffer_size)
        # if self.latent_dim is not None:
        #     self.register_buffer('buffer', torch.zeros(( [*self.latent_dim, self.buffer_size])))
        if bf16:
            self.buffer = torch.zeros(( [*self.latent_dim, self.buffer_size]), dtype=torch.bfloat16)
        else:
            self.buffer = torch.zeros(( [*self.latent_dim, self.buffer_size]), dtype=torch.float32)

    # 1D: (C, T) scheme, 2D: (C,H,W,T) scheme -> (Latent_DIM, T) scheme, T for time axis
    # Input: (C, 1) or (C, H, W, 1) -> point-by-point along time
    # Return: (C, L) or (C, H, W, L), L for latent state
    def forward(self, x: torch.Tensor):
        # if not hasattr(self, "buffer"):
        #     self.latent_dim = x.shape[0:-1]
        #     self.register_buffer('buffer', torch.zeros(( [*self.latent_dim, self.buffer_size]), dtype=x.dtype, device=x.device))
        self.buffer[..., 0:-1] = self.buffer[..., 1:]
        # print(self.buffer.shape, x.shape)
        self.buffer[..., -1] = x[..., -1]
        return self.buffer.to(torch.float32)