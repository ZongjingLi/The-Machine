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

    im = torch.randn([2,3,32,32])
    EBML = EBMLearner(config)

    answers = ["True","True"]
    outputs = EBML.ground_concept(im,["exist(scene())","exist(scene())"])
    print(outputs)
    print(ground_results(outputs,answers))