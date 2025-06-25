import torch
import torch.nn as nn
from .SSMOperator import SSMOperator


def ops_to_ssm(op_mod, latent_dim, num_latent_state=None, time_stride=None, ssm_cls=SSMOperator):
    if num_latent_state is None or time_stride is None:
        num_latent_state, time_stride = inference_ssm_params(op_mod)
    
    return ssm_cls(op_mod, num_latent_state, latent_dim, time_stride)

def inference_ssm_params(op_mod):
    num_s = None
    stride = None
    if isinstance(op_mod, nn.Conv1d):
        num_s = (op_mod.kernel_size[0] - 1) * op_mod.dilation[0] + 1
        stride = op_mod.stride[0]
    elif isinstance(op_mod, nn.AvgPool1d | nn.MaxPool1d):
        num_s = op_mod.kernel_size
        stride = op_mod.stride
    elif isinstance(op_mod, nn.Flatten):
        if (op_mod.end_dim == -1):
            raise NotImplementedError("Inference SSM Params: Flatten with end_dim=-1")
        num_s = 1
        stride = 1
    
    else:
        raise NotImplementedError("Inference SSM Params not supported for", type(op_mod))
    
    return num_s, stride



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