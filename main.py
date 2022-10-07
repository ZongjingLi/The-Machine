import torch
import torch.nn as nn

if __name__ == "__main__":
    from dataloader import *
    from perception_modules.savis import *
    parser = SlotAttentionParser(6,100,3)
    dataset = SpriteData("train")
    train_loader = DataLoader(dataset,batch_size = 2)
