import torch
import torch.nn as nn
import torch.fx as fx

from .GraphAnalyser import GraphAnalyser, SSMMetadata

from .OpsToSSM import ops_to_ssm

from .utils import get_graph_node_by_target