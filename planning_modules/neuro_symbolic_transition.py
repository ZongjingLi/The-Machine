import torch
import torch.nn as nn

class NSAbstractModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        return x

    def abstract(state,soft = False):
        return 0