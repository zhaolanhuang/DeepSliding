import torch
import torch.nn as nn
import torch.fx as fx

from functools import reduce

from Uncausal import is_uncausal_module, is_uncausal_function, is_uncausal_method

from typing import NamedTuple, Optional

from dataclasses import dataclass

from utils import get_graph_node_by_name

@dataclass
class SSMMetadata:
    is_causal: bool
    is_causal_breaker: bool
    is_global_pooling: bool
    tensor_meta_time_step: Optional[fx.passes.shape_prop.TensorMetadata]


# TODO: suport iteratively analyse customize modules inside the model


def find_uncausal_nodes(traced_graph, named_modules):
    nodes = traced_graph.nodes
    uncausal_nodes = []
    for n in nodes:
        if n.op == "call_module":
            called_mod = named_modules[n.target]
            if is_uncausal_module(called_mod):
                uncausal_nodes.append(n)
        elif n.op == "call_function":
            if is_uncausal_function(n.target):
                uncausal_nodes.append(n)
        elif n.op == "call_method":
            if is_uncausal_method(n.target):
                uncausal_nodes.append(n)
    return uncausal_nodes

def get_output_node_of(traced_graph):
    nodes = traced_graph.nodes
    for n in nodes:
        if n.op == "output":
            return n

# Return including the input node itself            
def get_all_nodes_from_graph_rely_on(node, traced_graph):
    output_node = get_output_node_of(traced_graph)
    relied_nodes = list()
    def _get_all_nodes_reverse_from(n, node_lst=[]):
        nonlocal relied_nodes
        if n is node:
            return node_lst
        inp_nodes = n.all_input_nodes
        if inp_nodes == []:
            return []
        else:
            node_lst = [*node_lst, *inp_nodes]
            for _n in inp_nodes:
                _nodes = _get_all_nodes_reverse_from(_n, node_lst)
                relied_nodes = [*relied_nodes, *_nodes]
            return []
    _get_all_nodes_reverse_from(output_node)
    return relied_nodes



class GraphAnalyser:
    # torch_mod: module
    # input_shape: (..., T), T as window size on time dimension
    # time_step_size: step size of siding window on time dimension
    def __init__(self, torch_mod: nn.Module, input_shape, time_step_size: int):
        self._named_modules = dict(torch_mod.named_modules())
        print(self._named_modules)
        self._traced_mod = fx.symbolic_trace(torch_mod.eval())
        self._traced_mod_time_step = fx.symbolic_trace(torch_mod.eval()) # traced graph for catching time step info, used for transformation
        self._traced_mod.graph.print_tabular()
        self._uncausal_nodes = find_uncausal_nodes(self._traced_mod.graph, self._named_modules)
        self._uncausal_nodes = self.get_all_uncausal_successors_of_uncausal_nodes()

        self.init_ssm_metadata_of_nodes()
        self.mark_causality_of_nodes()

        self._input_shape = input_shape
        self._time_step_size = time_step_size

        self.capture_complete_shape_info()

        self.capture_shape_info_with_time_step_size()

        self._causal_breaker = self.find_causal_breaker()

        print(self._causal_breaker)

        self.mark_global_pooling_of_nodes()

    
    @property
    def traced_mod(self):
        return self._traced_mod

    @property
    def traced_mod_with_time_step(self):
        return self._traced_mod_time_step

    @property
    def traced_graph(self):
        return self._traced_mod.graph

    @property
    def traced_graph_with_time_step(self):
        return self._traced_mod_time_step.graph
    
    @property
    def causal_breaker(self):
        return self._causal_breaker
    
    @property
    def named_modules(self):
        return self._named_modules
    
    def init_ssm_metadata_of_nodes(self):
        for n in self._traced_mod.graph.nodes:
            n.meta["ssm_meta"] = SSMMetadata(False, False, False, None)

    # find the nodes that breaks causality from the uncausal nodes
    def find_causal_breaker(self):
        causal_breaker = []
        for n in self._uncausal_nodes:
            inp_causality = [x.meta['causal'] for x in  n.all_input_nodes]
            is_all_inp_causal = reduce(lambda x,y: x and y, 
                                        inp_causality)
            if is_all_inp_causal:
                causal_breaker.append(n)
                n.meta['ssm_meta'].is_causal_breaker = True
        return causal_breaker



    def capture_complete_shape_info(self):
        sample_input = torch.randn(*self._input_shape)
        fx.passes.shape_prop.ShapeProp(self._traced_mod).propagate(sample_input)
        for node in self._traced_mod.graph.nodes:
            print(node.name, node.meta['tensor_meta'].dtype,
                node.meta['tensor_meta'].shape, node.meta['causal'])

    def capture_shape_info_with_time_step_size(self):

        sample_input = torch.randn(*self._input_shape[:-1], self._time_step_size)

        try:
            fx.passes.shape_prop.ShapeProp(self._traced_mod_time_step).propagate(sample_input)
        except RuntimeError as e:
            print("Stop by Ops needed for complete input shape", str(e))
        finally:
            pass

        for node in self._traced_mod_time_step.graph.nodes:
            if "tensor_meta" in node.meta:
                print(node.name, node.meta['tensor_meta'].dtype,
                    node.meta['tensor_meta'].shape)
                ori_node = get_graph_node_by_name(self._traced_mod.graph, node.name)
                ori_node.meta['ssm_meta'].tensor_meta_time_step = node.meta['tensor_meta']
            else:
                node.meta['tensor_meta'] = None



    def get_all_uncausal_successors_of_uncausal_nodes(self):
        nodes = []
        for un in self._uncausal_nodes:
            nodes += get_all_nodes_from_graph_rely_on(un, self._traced_mod.graph)
        return list(set(nodes))

    def mark_causality_of_nodes(self):
        for n in self._traced_mod.graph.nodes:
            if n in self._uncausal_nodes:
                n.meta['causal'] = False
                n.meta['ssm_meta'].is_causal = False
            else:
                n.meta['causal'] = True
                n.meta['ssm_meta'].is_causal = True

    def mark_global_pooling_of_nodes(self):
        for n in self._traced_mod.graph.nodes:
            if n.op != "call_module":
                continue
            op_mod = self._named_modules[n.target]
            if isinstance(op_mod, nn.AvgPool1d | nn.MaxPool1d):
                input_s = n.meta['tensor_meta'].shape[-1]
                kernel_size = op_mod.kernel_size
                if input_s == kernel_size:
                    n.meta['ssm_meta'].is_global_pooling = True
                else:
                    n.meta['ssm_meta'].is_global_pooling = False



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

    graph_analyser = GraphAnalyser(MyModel(), [40, 40], 10)