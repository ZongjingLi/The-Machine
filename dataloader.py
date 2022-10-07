import torch
import torch.nn as nn

from torch.utils.data import Dataset, dataloader

class SpriteData(Dataset):
    def __init__(self,split = "train"):
        super().__init__()

        assert split in ["train","test","val"],print("Unknow dataset split type: {}".format(split))
    def __len__(self):return 0

    def __getitem__(self,index):
        return 