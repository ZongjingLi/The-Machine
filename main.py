import torch
import torch.nn as nn

if __name__ == "__main__":
    from config import *
    from machine_prototypes.object_centric_learner import *
    LORL = Lorl(opt = config)