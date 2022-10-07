import torch
import torch.nn as nn

from torch.utils.data import Dataset, dataloader

class SpriteData(Dataset):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        return x