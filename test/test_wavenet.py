import torch
import torch.nn as nn

from ..GraphTransformer import GraphTransformer
from ..GraphAnalyser import GraphAnalyser

from ..model.WaveNet import DEFAULT_INPUT_SHAPE, DEFAULT_SLIDING_STEP_SIZE, WaveNet

if __name__ == "__main__":
    x = torch.randn(*DEFAULT_INPUT_SHAPE[:-1], 1)
    ori_mod = WaveNet().eval()
    graph_analyser = GraphAnalyser(ori_mod, DEFAULT_INPUT_SHAPE, 1)
    graph_transformer = GraphTransformer(graph_analyser, True)
    new_g = graph_transformer.transform()
    print(new_g)

    scripted_model = torch.jit.trace(new_g, x, check_trace=True).eval()
    print(scripted_model.inlined_graph)
    scripted_model.save("./DeepSliding/test/wavenet.pth")