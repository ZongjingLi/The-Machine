import torch
import torch.nn as nn

class NSAbstractModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.unary_predicates  = None
        self.binary_predicates = None

    def forward(self,x):
        return x

    def abstract(state,soft = False):
        # an abstract state is a diction with:
        # - a set of entities with center and offsets 1e-6
        # - a nullary entity that encodes some global info.
        return 0