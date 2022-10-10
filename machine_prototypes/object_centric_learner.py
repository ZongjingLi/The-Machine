import torch
import torch.nn as nn

import matplotlib.pyplot as plt
# namomo
from moic.data_structure import *
# this is the machine
class Lorl(nn.Module):
    def __init__(self,opt):
        super().__init__()
        self.visual_scene_parser = None
        self.program_parser = None
        self.knowledgebase = None
    
    def perception(self,image):
        return self.visual_scene_parser(image)

    def forward(self,x):
        return x

    def ground_concept(self,image,program):
        if isinstance(program,str):program = toFuncNode(program)
        return program