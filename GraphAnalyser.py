import torch
import torch.nn as nn
import torch.fx as fx

# TODO: suport iteratively analyse customize modules inside the model


def find_uncausal_nodes(traced_graph):
    nodes = traced_graph.nodes
    for n in nodes:
        print(n.name)

class GraphAnalyser:
    # torch_mod: module
    # input_shape: (..., T), T as window size on time dimension
    # time_step_size: step size of siding window on time dimension
    def __init__(self, torch_mod: nn.Module, input_shape, time_step_size: int):
        self._named_modules = dict(torch_mod.named_modules())
        print(self._named_modules)
        self._traced_mod = fx.symbolic_trace(torch_mod.eval())
        self._traced_mod.graph.print_tabular()
        find_uncausal_nodes(self._traced_mod.graph)
        self._input_shape = input_shape
        self._time_step_size = time_step_size




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
            self.conv1 = nn.Conv1d(4, 4, kernel_size=3)
            self.relu = nn.ReLU()
            self.conv2 = nn.Conv1d(4, 4, kernel_size=3)
            self.conv3 = nn.Conv1d(4, 4, kernel_size=3)
            self.flatten1 = nn.Flatten(0, -1)
            self.linear1 = nn.Linear(40,10)
            self.called_mod = CalledModule()

        def forward(self, x):
            x = self.conv1(x)
            x = self.relu(x)
            x = self.called_mod(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.flatten1(x)
            x = self.linear1(x)
            return x

    graph_analyser = GraphAnalyser(MyModel(), None, None)