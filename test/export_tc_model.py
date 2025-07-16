import torch
import torch.nn as nn

from ..GraphTransformer import GraphTransformer
from ..GraphAnalyser import GraphAnalyser

def export_tc_model(export_path, model_cls, input_shape, sliding_step_size):
    model_name = model_cls.__name__

    x = torch.randn(*input_shape[:-1], 1)
    ori_mod = model_cls().eval()
    graph_analyser = GraphAnalyser(ori_mod, input_shape, sliding_step_size)
    graph_transformer = GraphTransformer(graph_analyser, True)
    new_g = graph_transformer.transform()

    scripted_model = torch.jit.trace(new_g, x, check_trace=True).eval()
    scripted_model.save(f"{export_path}/ssm_{model_name}.pth")

    x = torch.randn(*input_shape)
    ori_scripted_model = torch.jit.trace(ori_mod, x, check_trace=True).eval()
    ori_scripted_model.save(f"{export_path}/{model_name}.pth")