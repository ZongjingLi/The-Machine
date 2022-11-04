import torch
import torch.nn as nn

import matplotlib.pyplot as plt

import numpy as np

import moic.utils  as utils

from moic.data_structure import *

class NeRFVAE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):return x