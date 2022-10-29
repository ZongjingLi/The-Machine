import torch
import torch.nn as nn

def ground_results(results,answers):
    outputs = []
    for i in range(len(results)):
        assert answers[i] in results[i]["outputs"]
        index_answer = results[i]["outputs"].index(answers[i])
        outputs.append(results[i]["scores"][index_answer])
    return outputs

if __name__ == "__main__":
    from config import *

    from machine_prototypes.ebm_learner import *

    im = torch.randn([3,3,64,64])
    EBML = EBMLearner(config)

    answers = ["True","True"]
    outputs = EBML.ground_concept(im,["exist(scene())","exist(filter(scene(),red))","count(scene())"])
    print(outputs)
    #print(ground_results(outputs,answers))