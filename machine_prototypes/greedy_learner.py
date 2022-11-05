import torch
import torch.nn as nn

import torch.nn.functional as F

from reasoning_modules.box_embedding_space import *
from parser_modules.vanilla_parser import *
from perception_modules.region_proposals import *
from moic.data_structure import *
from moic.utils import *

class GreedyLearner(nn.Module):
    def __init__(self,config,concepts):
        super().__init__()

        # 1. the energy based perception module, input the image and output the 
        self.component_model = None

        # 2. the symbolic program parser from language to program
        rule_diction = load_json(config.grammar)
        corpus       = []
        with open(config.corpus,"r") as corpus_file:
            for line in corpus_file:corpus.append(line.strip())
        # make the vanilla program parser made by ZongjingLi
        self.program_parser = make_program_parser(corpus,rule_diction)

        # 3. quasi-symbolic concept program executor
        if config.concepts is not None:concepts = torch.load(config.concepts)
        self.quasi_executor = QuasiExecutor(concepts)
        self.latents = None

    def perception(self,im):
        latents = self.component_model(im)
        return list(latents)
    
    def ground_concept(self,image,programs,answers = None):
        if isinstance(programs[0],str): programs = [toFuncNode(program) for program in programs]
        latents = self.perception(image)
        self.latents = latents
        results = [];eps = 1e-4
        for i in range(latents[0].shape[0]):
            features = torch.stack([latent[i:i+1] for latent in latents])

            context = {"features":cast_to_entities(features),"scores":-eps - torch.zeros([len(latents)])}
            results.append(self.quasi_executor(programs[i],context))
        return results

    def generate_concepts(concepts):
        outputs = []
        for concept in concepts:
            pass
        return outputs

    def parse_concept(self,concept):
        return 0

    def __repr__(self):return "Violet Evergarden"

    def forward(self,x):
        return x

    def __str__(self):return "the name is Violet Evergarden"