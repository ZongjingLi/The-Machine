import torch
import torch.nn as nn

from melkor_parser import *

rule_diction = {
    "filter":{"input_types":["object","concept"]},
    "scene":{"input_types":[]},
    "red":{"input_types":None},
}

machine_cfg     = ContextFreeGrammar(rule_diction,64)
program_decoder = Decoder(128,64,132,machine_cfg)

parser = LanguageParser(64,128,program_decoder,corpus = ["what is that"])

outputs,lp = parser("what is that")
print(outputs,lp)
