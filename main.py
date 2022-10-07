import torch
import torch.nn as nn

if __name__ == "__main__":
    from perception_modules.savis import *
    parser = SlotAttentionParser(6,100,3)