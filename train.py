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

def train_ebml(model,dataset,joint = False,visualize = False):
    from tqdm import tqdm
    optimizer = torch.optim.Adam(model.parameters(),lr = 2e-4)

    trainloader = DataLoader(dataset)
    
    for epoch in tqdm(range(1000)):
        total_loss = 0
        for sample in trainloader:
            # collect data from the sample
            images    = sample["image"];questions = sample["question"]
            programs  = sample["program"];answers = sample["answer"]
            
            # ground the concept in the image by neuro-symbolic programs
            results = model.ground_concept(images,programs)
            
            # ground the results from the quasi-symbolic executor
            logprobs  = ground_results(results,answers)
            
            # add all the terms from the quasi-symbolic executor to the qa-loss
            qa_loss = 0
            for term in logprobs:qa_loss - term
            
            if joint:
                energy_loss = 0;recons_loss = 0
            # if it is a joint training, add the reconstruction loss and energy loss from the ebm

            # optimize the concept structure by reduce the the qa-loss Pr[a,e(c,im)]
            qa_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if joint:
            print("epoch: {} qa_loss:{} recons: {} energy_loss:{}".format(epoch,qa_loss,recons_loss,energy_loss))
        else:
            print("epoch: {} qa_loss:{}".format(epoch,qa_loss))

if __name__ == "__main__":
    from config import *
    from machine_prototypes.ebm_learner import *

    # create the energy-based meta-concept learner
    EBML = EBMLearner(config)

    train_ebml(EBML,[],joint = False)
