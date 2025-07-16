import torch
import torch.nn as nn

from ..GraphTransformer import GraphTransformer
from ..GraphAnalyser import GraphAnalyser

from .export_tc_model import export_tc_model

from ..model.CET import CET_S


if __name__ == "__main__":
    export_tc_model("./DeepSliding/test/", CET_S, CET_S.DEFAULT_INPUT_SHAPE, CET_S.DEFAULT_SLIDING_STEP_SIZE)
# if __name__ == "__main__":
#     x = torch.randn(*DEFAULT_INPUT_SHAPE[:-1], 1)
#     ori_mod = CET_S().eval()
#     graph_analyser = GraphAnalyser(ori_mod, DEFAULT_INPUT_SHAPE, DEFAULT_SLIDING_STEP_SIZE)
#     graph_transformer = GraphTransformer(graph_analyser, True)
#     new_g = graph_transformer.transform()
#     graph_analyser.traced_graph.print_tabular()
#     # new_g.graph.print_tabular()

#     scripted_model = torch.jit.trace(new_g, x, check_trace=True).eval()
#     scripted_model.save("./DeepSliding/test/ssm_CET.pth")
#     # print(scripted_model)

#     x = torch.randn(*DEFAULT_INPUT_SHAPE)
#     ori_scripted_model = torch.jit.trace(ori_mod, x, check_trace=True).eval()
#     ori_scripted_model.save("./DeepSliding/test/CET.pth")
#     # print(ori_scripted_model)