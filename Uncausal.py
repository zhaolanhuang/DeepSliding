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

    #
    "ChannelShuffle",

]

def _is_uncausal_softmax(mod):
    if mod.dim == -1:
        return True
    else:
        return False
    
# [name: judge func(mod)]
COULD_BE_UNCAUSAL_MODULES = {
    "Softmin": _is_uncausal_softmax,
    "Softmax": _is_uncausal_softmax,
    "Softmax2d": _is_uncausal_softmax,
    "LogSoftmax": _is_uncausal_softmax,

}
    
def is_uncausal_module(mod):
    if isinstance(mod, str):
        return (mod in UNCAUSAL_MODULE_NAMES)
    elif isinstance(mod, nn.Module):
        if (type(mod).__name__ in COULD_BE_UNCAUSAL_MODULES):
            return COULD_BE_UNCAUSAL_MODULES[type(mod).__name__](mod)
        return type(mod).__name__ in UNCAUSAL_MODULE_NAMES
    else:
        raise TypeError(f"Unreconized module type: {type(mod)}")



UNCAUSAL_FUNCTION_NAMES = [
    
]




# TODO: Impl
def is_uncausal_function(func):
    return False

# TODO: Impl
def is_uncausal_method(method):
    if method == "transpose": #TODO: temporary fix for ResTCN
        return True
    elif method == "squeeze": #TODO: temprorary fix for TinyChirpTime
        return True
    return False