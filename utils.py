import torch
import torch.fx as fx

def get_graph_node_by_target(graph: fx.Graph, target):
    for n in graph.nodes:
        if n.target == target:
            return n
    raise RuntimeError(f"Cannot find graph node by name with: {target}")
        
def get_graph_node_by_name(graph: fx.Graph, name):
    for n in graph.nodes:
        if n.name == name:
            return n
    raise RuntimeError(f"Cannot find graph node by name with: {name}")