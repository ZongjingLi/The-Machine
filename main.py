import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

def ground_results(results,answers):
    outputs = []
    for i in range(len(results)):
        # discrete output probability distribution
        if isinstance(results[i],dict):
            assert answers[i] in results[i]["outputs"]
            index_answer = results[i]["outputs"].index(answers[i])
            outputs.append(results[i]["scores"][index_answer])
        # continuous output, ground with additional prior
        else:
            target_answer = int(answers[i])
            outputs.append( torch.log(torch.sigmoid((0.5-torch.abs(results[i] - target_answer) )/0.125)))
    return outputs

def train_ebml(model,dataset,joint = False):
    from tqdm import tqdm
    optimizer = torch.optim.Adam(model.parameters(),lr = 2e-4)

    trainloader = DataLoader(dataset)
    
    for epoch in tqdm(range(1000)):
        total_loss = 0
        for sample in trainloader:
            images    = sample["image"]
            questions = sample["question"]
            programs  = sample["program"]
            answers   = sample["answer"]
            
            results = model.ground_concept(images,programs)

            logprobs  = ground_results(results,answers)
            
            qa_loss = 0
            for term in logprobs:qa_loss - term
            
            qa_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        print()

if __name__ == "__main__":
    from config import *

    from machine_prototypes.ebm_learner import *

    im = torch.randn([3,3,64,64])
    EBML = EBMLearner(config)
    
    answers = ["True","True","3"]
    outputs = EBML.ground_concept(im,["exist(scene())","exist(filter(scene(),blue))","count(scene())"])
    print(outputs)
    print(ground_results(outputs,answers))