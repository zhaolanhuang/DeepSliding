import torch
import torch.nn as nn

from ..GraphTransformer import GraphTransformer
from ..GraphAnalyser import GraphAnalyser

from .export_tc_model import export_tc_model

from ..model.CET import CET_S


if __name__ == "__main__":
    export_tc_model("./DeepSliding/test/", CET_S, CET_S.DEFAULT_INPUT_SHAPE, CET_S.DEFAULT_SLIDING_STEP_SIZE)
