import sys
import os
sys.path.append(os.getenv("TVM_HOME") + '/python')

import tvm
from tvm import relay
from tvm.ir import IRModule
import tvm.contrib.utils
from tvm.micro import export_model_library_format
from tvm.driver import tvmc
from tvm.driver.tvmc.frontends import guess_frontend, PyTorchFrontend
import inspect

from .SSMFakeOperator import SSMFakeOperator
from .tvm_op.ssm import ssm

def ssm_fake_op(inputs, input_types):
    data = inputs[0]
    num_latent_states = inputs[1]
    latent_dim = tuple(inputs[2])
    stride = inputs[3]
    return ssm(data, num_latent_states, latent_dim, stride)

custom_convert_map = {
    "DeepSliding::ssm_fake_op": ssm_fake_op,
}

def load_model(model_path: str, shape_dict=None):
    frontend = guess_frontend(model_path)
    if isinstance(frontend, PyTorchFrontend):
        if 'preserve_pytorch_scopes' in inspect.getfullargspec(relay.frontend.from_pytorch).args:
            model = tvmc.load(model_path, shape_dict=shape_dict, 
                            use_parser_friendly_name=True, preserve_pytorch_scopes=True,
                            custom_convert_map=custom_convert_map
                            )
        else:
            model = tvmc.load(model_path, shape_dict=shape_dict, 
                            use_parser_friendly_name=True, custom_convert_map=custom_convert_map
                                )
    else:
        raise NotImplementedError()
    print(model.mod)
    return model.mod, model.params

if __name__ == "__main__":
    mod, params = load_model('./DeepSliding/torchscrpited_model.pth', {'input': (50, 1)})
    print(mod)