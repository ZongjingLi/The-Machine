import torch
import torch.nn as nn
from   perception_modules.energy_base import *

import argparse

model_parser = argparse.ArgumentParser()
model_config = model_parser.parse_args(args = [])

class EBMLearner(nn.Module):
    def __init__(self,config):
        super().__init__()

        # 1. the energy based perception module, input the image and output the 


        # 2. the symbolic program parser from language to program
        
        # 3. quasi-symbolic concept program executor


    def forward(self,x):
        return x