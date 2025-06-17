import sys
import os
sys.path.append(os.getenv("TVM_HOME") + '/python')

import tvm
from tvm.ir import IRModule

custom_convert_map = {
    "DeepSliding::ssm_fake_op": ssm_fake_op,
}

def ssm_fake_op(inputs, input_types):
    data = inputs[0]
    num_latent_states = inputs[1]
    latent_dim = tuple(inputs[2])
    stride = input[3]