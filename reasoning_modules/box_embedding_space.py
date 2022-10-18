import torch
import torch.nn as nn

from melkor_knowledge import *

blue = ConceptBox("blue","color")
red  = ConceptBox("red" ,"color")

e1   = EntityBox(torch.randn([1,100]))


lp = logJointVolume(e1,blue,True)

print(lp)