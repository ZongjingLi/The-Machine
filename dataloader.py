import torch
import torch.nn as nn

import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from moic.utils import load_json
from PIL import Image

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
    
        self.split = split
        self.root_dir = "datasets/sprites3"
        self.files = os.listdir(os.path.join(self.root_dir,split))
        self.img_transform = transforms.Compose(
            [transforms.ToTensor()]
        )

        self.questions = load_json("datasets/sprites3/train_sprite3_qa.json")
        
    def __len__(self): return 100#len(self.files)

    def __getitem__(self,index):
        path = self.files[index]
        # open the image of the file
        image = Image.open(os.path.join(self.root_dir,self.split,"{}_{}.png".format(self.split,index)))
        image = image.convert("RGB").resize([64,64])
        image = self.img_transform(image)

        sample = {"image":image,"question":None,"program":None,"answer":None}
        return sample