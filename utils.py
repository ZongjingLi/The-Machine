import torch
import torch.nn as nn

from moic.data_structure import *

import networkx as nx

def toNxGraph(func):
    nodes = []
    def break_node(node):
        nodes.append(node.token)
        for c in node.children:break_node(c)
    break_node(func)
    print(nodes)

func = toFuncNode("exist(filter(scene(),red))")

func_g = toNxGraph(func)
