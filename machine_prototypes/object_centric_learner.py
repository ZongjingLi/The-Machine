import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from moic.data_structure import *

from parser_modules.vanilla_parser import make_program_parser
from reasoning_modules.box_embedding_space import *
from perception_modules.savis import *

# this is the machine
class Lorl(nn.Module):
    def __init__(self,opt):
        super().__init__()
        self.visual_scene_parser = None
        self.program_parser = make_program_parser()
        self.knowledgebase = None
    
    def perception(self,image):
        return self.visual_scene_parser(image)
        # namomo

    def forward(self,x):
        return x

    def ground_concept(self,image,program):
        if isinstance(program,str):program = toFuncNode(program)
        return program