import torch
import torch.nn as nn

import matplotlib.pyplot as plt

from moic.data_structure import *

class Lorl(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,x):
        return x

    def ground_concept(self,image,program):
        if isinstance(program,str):program = toFuncNode(program)
        return program