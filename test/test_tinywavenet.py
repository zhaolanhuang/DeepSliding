import torch
import torch.nn as nn

from ..GraphTransformer import GraphTransformer
from ..GraphAnalyser import GraphAnalyser

from ..model.WaveNet import TinyWaveNet

if __name__ == "__main__":
    x = torch.randn(*TinyWaveNet.DEFAULT_INPUT_SHAPE[:-1], 1)
    ori_mod = TinyWaveNet().eval()
    graph_analyser = GraphAnalyser(ori_mod, TinyWaveNet.DEFAULT_INPUT_SHAPE, TinyWaveNet.DEFAULT_SLIDING_STEP_SIZE)
    graph_transformer = GraphTransformer(graph_analyser, True)
    new_g = graph_transformer.transform()
    new_g.graph.print_tabular()

    scripted_model = torch.jit.trace(new_g, x, check_trace=True).eval()
    scripted_model.save("./DeepSliding/test/ssm_tinywavenet.pth")

    x = torch.randn(*TinyWaveNet.DEFAULT_INPUT_SHAPE)
    ori_scripted_model = torch.jit.trace(ori_mod, x, check_trace=True).eval()
    ori_scripted_model.save("./DeepSliding/test/tinywavenet.pth")