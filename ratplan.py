import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from melkor_logic.NLM.model import *

from config import *

struct_config = ((3 ,12,3 ),
                 (30,50,20),
                 (20,30,3 ),
                 (1 ,30,3 ))

class RatSkill(nn.Module):
    def __init__(self,name = "default_name"):
        super().__init__()
        self.name = name
        self.start_condition = NeuroLogicMachine(struct_config)
        self.end_condition = NeuroLogicMachine(struct_config)

        self.policy_net = None

    def isStart(self,state):return self.start_condition(state)

    def isEnd(self,state):return self.end_condition(state)

class FiniteStateConstructor(nn.Module):
    def __init__(self):
        super().__init__()

    def construct(self,inputs):
        # function used to constuct the finite state machine.
        return inputs

class RRT(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):return x

class AStartSearch(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):return x

skill = RatSkill()
