import torch
import torch.nn as nn

if __name__ == "__main__":
    from config import *
    from machine_prototypes.object_centric_learner import *
    from machine_prototypes.ebm_learner import *
    im = torch.randn([2,3,64,64])
    EBML = EBMLearner(config)

    outputs = EBML.ground_concept(im,["exist(scene())","exist(scene())"])
    print(outputs)