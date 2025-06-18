import torch
import torch.nn as nn

from GraphTransformer import GraphTransformer
from GraphAnalyser import GraphAnalyser


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
            self.flatten1 = nn.Flatten(0, -1)
            self.linear1 = nn.Linear(136,10)
            self.called_mod = CalledModule()

        def forward(self, x):
            x = self.conv1(x)
            x2 = self.relu(x)
            # x = self.called_mod(x2) # TODO: this place cause bug inside tvm, conv1 <-> called_mod.conv1 verwechseln...
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.flatten1(x)
            x1 = x * 2
            x += x1
            x = self.linear1(x)
            return x
    x = torch.randn(50, 1)
    ori_mod = MyModel().eval()
    graph_analyser = GraphAnalyser(ori_mod, [50, 40], 10)
    graph_transformer = GraphTransformer(graph_analyser, True)
    new_g = graph_transformer.transform()
    print(new_g)

    scripted_model = torch.jit.trace(new_g, x, check_trace=True).eval()
    print(scripted_model.inlined_graph)
    scripted_model.save("torchscrpited_model.pth")