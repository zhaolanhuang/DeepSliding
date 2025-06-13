import torch
import torch.nn as nn
import torch.fx as fx

from GraphAnalyser import GraphAnalyser

from utils import get_graph_node_by_target

SSMABLE_OP_NAMES = [
    "Conv1D",
    "Average"
]

class GraphTransformer(fx.Transformer):
    def __init__(self, torch_mod: nn.Module, input_shape, time_step_size: int):
        
        self._analyser = GraphAnalyser(torch_mod, input_shape, time_step_size)
        self._traced_mod = self._analyser.traced_mod
        self._traced_mod_ts = self._analyser.traced_mod_with_time_step
        self._causal_breaker = self._analyser.causal_breaker
        super().__init__(self._traced_mod)

    def call_function(self, target, args, kwargs):
        return super().call_function(target, args, kwargs)
    
    def call_method(self, target, args, kwargs):
        return super().call_method(target, args, kwargs)
    
    def call_module(self, target, args, kwargs):
        node: fx.Node = get_graph_node_by_target(self._traced_mod.graph, target)
        print(node.meta['ssm_meta'])
        return super().call_module(target, args, kwargs)
    


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
            self.conv1 = nn.Conv1d(40, 4, kernel_size=3)
            self.relu = nn.ReLU()
            self.conv2 = nn.Conv1d(4, 4, kernel_size=3)
            self.conv3 = nn.Conv1d(4, 4, kernel_size=3)
            self.flatten1 = nn.Flatten(0, -1)
            self.linear1 = nn.Linear(128,10)
            self.called_mod = CalledModule()

        def forward(self, x):
            x = self.conv1(x)
            x2 = self.relu(x)
            x = self.called_mod(x2)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.flatten1(x)
            x1 = x * 2
            x += x1
            x = self.linear1(x)
            return x

    graph_transformer = GraphTransformer(MyModel(), [40, 40], 10)
    graph_transformer.transform()