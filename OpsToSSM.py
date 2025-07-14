import torch
import torch.nn as nn
from .SSMOperator import SSMOperator


def ops_to_ssm(op_mod, in_shape, ssm_cls=SSMOperator):
    # if num_latent_state is None or time_stride is None:
    latent_dim, num_latent_state, time_stride = inference_ssm_params(op_mod, in_shape)
    
    return ssm_cls(op_mod, num_latent_state, latent_dim, time_stride)

def inference_ssm_params(op_mod, in_shape: tuple):
    latent_dim = in_shape[:-1]
    num_s = None
    stride = None
    if isinstance(op_mod, nn.Conv1d):
        num_s = (op_mod.kernel_size[0] - 1) * op_mod.dilation[0] + 1
        stride = op_mod.stride[0]
    elif isinstance(op_mod, nn.AvgPool1d | nn.MaxPool1d):
        num_s = op_mod.kernel_size if type(op_mod.kernel_size) is int else op_mod.kernel_size[0]
        stride = op_mod.stride if type(op_mod.stride) is int else op_mod.stride[0]
    elif isinstance(op_mod, nn.AdaptiveAvgPool1d | nn.AdaptiveMaxPool1d):
        raise NotImplementedError("Adaptive Pooling shall be rewritten to normal pooling.")
         
    elif isinstance(op_mod, nn.Flatten):
        if (op_mod.end_dim == -1):
            num_s = in_shape[-1]
        else:
            num_s = 1
        stride = 1
    elif isinstance(op_mod, nn.Linear):
        num_s = op_mod.in_features
        stride = 1
    else:
        raise NotImplementedError("Inference SSM Params not supported for", type(op_mod))
    
    return latent_dim, num_s, stride



def conv1d_to_ssm(mod):
    pass

def conv2d_to_ssm(mod):
    pass

def conv_to_ssm(mod):
    pass

def global_pooling_to_ssm(mod):
    pass

def pooling_to_ssm(mod):
    pass

def attention_to_ssm(mod):
    pass

def flatten_to_ssm(mod):
    pass