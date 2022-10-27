import torch
import torch.nn as nn
from  perception_modules.energy_base import *
from  parser_modules.vanilla_parser  import *
from  reasoning_modules.box_embedding_space import *

import argparse

model_parser = argparse.ArgumentParser()
model_config = model_parser.parse_args(args = [])

class EBMLearner(nn.Module):
    def __init__(self,config):
        super().__init__()

        # 1. the energy based perception module, input the image and output the 
        self.component_model = LatentEBM(ebm_config)
        self.ebm_models = [self.component_model for _ in range(ebm_config.components)]

        # 2. the symbolic program parser from language to program
        rule_diction = load_json(config.grammar)
        corpus       = []
        with open(config.corpus,"r") as corpus_file:
            for line in corpus_file:corpus.append(line.strip())
        # make the vanilla program parser made by ZongjingLi
        self.program_parser = make_program_parser(corpus,rule_diction)

        # 3. quasi-symbolic concept program executor
        self.quasi_executor = QuasiExecutor({"static_concepts" :[],
                                             "dynamic_concepts":[],
                                             "relations":[]})

    def perception(self,im):
        return im

    def ground_concept(self,image,program,answer):
        if isinstance(program,str): program = toFuncNode(program)
        return image

    def forward(self,x):
        return x