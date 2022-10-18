import torch
import torch.nn as nn

import matplotlib.pyplot as plt

class RatSkill(nn.Module):
    def __init__(self,config):
        super().__init__()