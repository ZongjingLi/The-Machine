import torch
import torch.nn as nn

from moic.data_structure import *

import networkx as nx

import matplotlib.pyplot as plt

def toNxGraph(func):
    nodes = []
    edges = []
    def break_node(node):
        nodes.append(node.token);
        for c in node.children:
            edges.append([node.token,c.token])
            break_node(c)
    break_node(func)
    func_graph = nx.DiGraph()
    func_graph.add_nodes_from(nodes)
    func_graph.add_edges_from(edges)
    return func_graph
    

func = toFuncNode("exist(filter(scene(),red))")

func_g = toNxGraph(func)

print(func_g)
nx.draw_spectral(func_g,with_labels = True)

plt.show()