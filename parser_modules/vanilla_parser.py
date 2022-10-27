import torch
import torch.nn as nn
from moic.utils    import save_json
from melkor_parser import *

def dfs_seq(program):
    if isinstance(program,str):program = toFuncNode(program)
    outputs = []
    def dfs(node):
        outputs.append(node.token)
        for c in node.children:dfs(c)
    dfs(program)
    while "" in outputs:outputs.remove("")
    return outputs

def make_program_parser(corpus,cfg_diction,word_dim=256,signal_dim=32,key_dim=42,latent_dim=132):
    cfg = ContextFreeGrammar(cfg_diction,signal_dim)
    decoder = Decoder(signal_dim,key_dim,latent_dim,cfg)
    model = LanguageParser(word_dim,signal_dim,decoder,corpus)
    return model

if __name__ == "__main__":
    rule_diction = {
    "filter":{"input_types":["object_set","concept"]},
    "scene":{"input_types":[]},
    "red":{"input_types":None},
    "exist":{"input_types":["object_set"]},
    }
    save_json(rule_diction,"grammar.json")
    parser =  make_program_parser(["what is that","is there any red object"],rule_diction)
    outputs,lp = parser("is there any red object",dfs_seq("exist(filter(scene(),red))"))
    print(outputs,lp)
