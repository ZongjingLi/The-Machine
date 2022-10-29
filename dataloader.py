import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

class SpriteData(Dataset):
    def __init__(self,split = "train"):
        super().__init__()
        assert split in ["train","test","val"],print("Unknown dataset split type: {}".format(split))
    def __len__(self):return 0

    def __getitem__(self,index):
        return 

class BattlecodeData(Dataset):
    def __init__(self,split = "train"):
        super().__init__()
        assert split in [],print("Unknown dataset split type: {}".format(split))


class Clevr4(Dataset):
    def __init__(self,split = "train"):
        super().__init__()
        assert split in ["train","test","validate"],print("Unknown dataset split type: {}".format(split))

    def len(self):return 1

    def __getitem__(self,index):
        return index

class Sprite3(Dataset):
    def __init__(self,split = "train"):
        super().__init__()
        assert split in ["train","test","validate"],print("Unknown dataset split type: {}".format(split))
    
    def len(self):return 1000
    
    def __getitem__(self,index):
        return index