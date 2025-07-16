import torch
import torch.nn as nn

from ..GraphTransformer import GraphTransformer
from ..GraphAnalyser import GraphAnalyser

from ..model.tinychirp_cnn_time import TinyChirpCNNTime

if __name__ == "__main__":
    x = torch.randn(*TinyChirpCNNTime.DEFAULT_INPUT_SHAPE[:-1], 1)
    ori_mod = TinyChirpCNNTime().eval()
    graph_analyser = GraphAnalyser(ori_mod, TinyChirpCNNTime.DEFAULT_INPUT_SHAPE, TinyChirpCNNTime.DEFAULT_SLIDING_STEP_SIZE)
    graph_transformer = GraphTransformer(graph_analyser, True)
    new_g = graph_transformer.transform()
    new_g.graph.print_tabular()

    scripted_model = torch.jit.trace(new_g, x, check_trace=True).eval()
    scripted_model.save("./DeepSliding/test/ssm_TinyChirpCNNTime.pth")

    x = torch.randn(*TinyChirpCNNTime.DEFAULT_INPUT_SHAPE)
    ori_scripted_model = torch.jit.trace(ori_mod, x, check_trace=True).eval()
    ori_scripted_model.save("./DeepSliding/test/TinyChirpCNNTime.pth")