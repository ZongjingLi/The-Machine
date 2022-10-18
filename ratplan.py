import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from melkor_logic.NLM.model import *

from config import *

class RatSkill(nn.Module):
    def __init__(self,name = "default_name"):
        super().__init__()
        self.name = name
        self.start_condition = NeuroLogicMachine()
        self.end_condition = NeuroLogicMachine()

skill = RatSkill()