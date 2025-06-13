import torch
import torch.fx as fx

def get_graph_node_by_target(graph: fx.Graph, target):
    for n in graph.nodes:
        if n.target == target:
            return n
        
def get_graph_node_by_name(graph: fx.Graph, name):
    for n in graph.nodes:
        if n.name == name:
            return n