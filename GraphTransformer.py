import torch
import torch.nn as nn
import torch.fx as fx

from .GraphAnalyser import GraphAnalyser, SSMMetadata

from .OpsToSSM import ops_to_ssm

from .utils import get_graph_node_by_target

from .SSMOperator import WaitForNextInputError, SSMOperator

from .SSMFakeOperator import SSMFakeOperator

SSMABLE_OP_NAMES = [
    "Conv1d",
    "AvgPool1d",
    "MaxPool1d",
    "Flatten",
    "AdaptiveMaxPool1d",
    "AdaptiveAvgPool1d"
]

def is_ssm_able(op_mod):
    if isinstance(op_mod, str):
        return (op_mod in SSMABLE_OP_NAMES)
    elif isinstance(op_mod, nn.Module):
        return type(op_mod).__name__ in SSMABLE_OP_NAMES
    else:
        raise TypeError(f"Unreconized module type: {type(op_mod)}")

class GraphTransformer(fx.Transformer):    
    def __init__(self, analyser: GraphAnalyser, is_fake_ssm=False):
        self._analyser = analyser
        self._traced_mod = self._analyser.traced_mod
        self._traced_mod_ts = self._analyser.traced_mod_with_time_step
        self._causal_breaker = self._analyser.causal_breaker
        if is_fake_ssm:
            self._ssm_op_cls = SSMFakeOperator
        else:
            self._ssm_op_cls = SSMOperator
        super().__init__(self._traced_mod)

    @classmethod
    def from_torch_module(cls, torch_mod: nn.Module, input_shape, time_step_size: int, is_fake_ssm=False):
        return cls(GraphAnalyser(torch_mod, input_shape, time_step_size), is_fake_ssm)

    def ops_to_ssm(self, op_mod, in_shape):
        return ops_to_ssm(op_mod, in_shape, self._ssm_op_cls)

    def call_function(self, target, args, kwargs):
        return super().call_function(target, args, kwargs)
    
    def call_method(self, target, args, kwargs):
        return super().call_method(target, args, kwargs)
    
    def call_module(self, target: str, args, kwargs):
        if target.endswith("_ssm"):
            return super().call_module(target, args, kwargs)
        node: fx.Node = get_graph_node_by_target(self._traced_mod.graph, target)
        ssm_meta : SSMMetadata = node.meta['ssm_meta']
        op_mod = self.fetch_attr(target)
        out_tensor_meta = node.meta['tensor_meta']
        in_tensor_meta = node.all_input_nodes[0].meta['tensor_meta']
        print(node.target, node.meta['ssm_meta'])


        # causal breaker is also causal
        # TODO: dont use concept of "causal" cause in time series the whole model is actually causal, use time-accumulative operator instead
        if (ssm_meta.is_causal or ssm_meta.is_causal_breaker) and is_ssm_able(op_mod):
            if ssm_meta.is_global_pooling:
                print("Found global Pooling!")
            if isinstance(op_mod, nn.AdaptiveAvgPool1d | nn.AdaptiveMaxPool1d): # rewrite Adaptive to non-adaptive
                pool_cls = nn.AvgPool1d if isinstance(op_mod, nn.AdaptiveAvgPool1d) else nn.MaxPool1d
                stride = (in_tensor_meta.shape[-1]//op_mod.output_size)  
                k = in_tensor_meta.shape[-1] - (op_mod.output_size-1)*stride
                op_mod = pool_cls(k, stride, 0)
            new_ssm_mod = self.ops_to_ssm(op_mod, in_tensor_meta.shape)
            new_name = target + "_ssm"
            new_name = new_name.replace(".", "_") # replace . with _ to avoide name confilcts in tvm
            self._traced_mod.add_submodule(new_name, new_ssm_mod)
            return self.call_module(new_name, args, kwargs)
        return super().call_module(target, args, kwargs)
    
    def inference_causal_breaker_ssm_params(self):
        pass
    


if __name__ == "__main__":

    class CalledModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv1d(4, 4, kernel_size=3)
        
        def forward(self, x):
            return self.conv1(x)

    class MyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv1d(50, 4, kernel_size=3)
            self.relu = nn.ReLU()
            self.conv2 = nn.Conv1d(4, 4, kernel_size=3)
            self.conv3 = nn.Conv1d(4, 4, kernel_size=3)
            self.global_pool = nn.AdaptiveAvgPool1d(1)
            self.flatten1 = nn.Flatten(0, -1)
            self.linear1 = nn.Linear(4,10)
            self.called_mod = CalledModule()

        def forward(self, x):
            x = self.conv1(x)
            x2 = self.relu(x)
            x = self.called_mod(x2)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.global_pool(x)
            x = self.flatten1(x)
            x1 = x * 2
            x += x1
            x = self.linear1(x)
            return x
    x = torch.randn(50, 40)
    ori_mod = MyModel()
    graph_transformer = GraphTransformer.from_torch_module(ori_mod, [50, 40], 40)
    new_g = graph_transformer.transform()
    
    y = ori_mod(x)
    for i in range(40):
        try:
            y_iter = new_g(x[..., i:i+1])
        except WaitForNextInputError as e:
            pass
        finally:
            pass
    print("y:", y)
    print("y_iter:", y_iter)
    abs_err = torch.dist(y, y_iter, 1)
    y_n = torch.norm(y, 1)
    print("abs_err:", abs_err)
    print("y_norm:", y_n)
    print("rel_err:", abs_err / y_n)
    assert torch.allclose(y, y_iter)
    print("Result allclose!")