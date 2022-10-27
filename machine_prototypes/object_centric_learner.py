import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from moic.utils          import save_json,load_json
from moic.data_structure import *

from parser_modules.vanilla_parser         import make_program_parser
from reasoning_modules.box_embedding_space import *
from perception_modules.savis              import *

# this is the machine
class Lorl(nn.Module):
    def __init__(self,opt):
        super().__init__()
        # create the visual perception module using the savis perception
        self.visual_scene_parser = None

        # open the rule diction of cfg and the corpus file
        rule_diction = load_json(opt.grammar)
        corpus       = []
        with open(opt.corpus,"r") as corpus_file:
            for line in corpus_file:corpus.append(line.strip())
        # make the vanilla program parser made by ZongjingLi
        self.program_parser = make_program_parser(corpus,rule_diction)
        
        # make the quasi-symbolic executor over a set of concepts
        self.knowledgebase  = None#QuasiExecutor(concepts)
    
    def perception(self,image):
        return self.visual_scene_parser(image)
        # namomo

    def forward(self,x):
        return x

    def ground_concept(self,image,program):
        if isinstance(program,str):program = toFuncNode(program)
        return program