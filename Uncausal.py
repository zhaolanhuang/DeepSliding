import torch
import torch.nn as nn

#record torch methods/functions/modules that break the causality
# TODO: or shall we rename it causal breaker instead?
UNCAUSAL_MODULE_NAMES = [
    
    #MLP on Time axis
    "Bilinear",
    "LazyLinear",
    "Linear",

    # Flatten
    "Flatten", # TODO: could be causal if end_dim != -1
    "Unflatten"

    #Padding
    "CircularPad1d",
    "CircularPad2d",
    "CircularPad3d",
    "ConstantPad1d",
    "ConstantPad2d",
    "ConstantPad3d",
    "ReflectionPad1d",
    "ReflectionPad2d",
    "ReflectionPad3d",
    "ReplicationPad1d",
    "ReplicationPad2d",
    "ReplicationPad3d",
    "ZeroPad1d",
    "ZeroPad2d",
    "ZeroPad3d",

    # Activations
    "MultiheadAttention",
    "Softmin",
    "Softmax",
    "Softmax2d",
    "LogSoftmax",

    #
    "ChannelShuffle",

]

UNCAUSAL_FUNCTION_NAMES = [
    
]

def is_uncausal_module(mod):
    if isinstance(mod, str):
        return (mod in UNCAUSAL_MODULE_NAMES)
    elif isinstance(mod, nn.Module):
        return type(mod).__name__ in UNCAUSAL_MODULE_NAMES
    else:
        raise TypeError(f"Unreconized module type: {type(mod)}")

# TODO: Impl
def is_uncausal_function(func):
    return False

# TODO: Impl
def is_uncausal_method(method):
    return False