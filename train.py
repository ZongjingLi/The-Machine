import torch
import torch.nn as nn
from dataloader import *
from torch.utils.data import Dataset, DataLoader
from utils import *

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

    trainloader = DataLoader(dataset,batch_size = 64,shuffle=True)
    loss_history = [];plt.ion()

    if joint:
        model.ebm_models = [model.component_model for _ in range(4)]
        ebm_optimizers = [torch.optim.Adam(model.parameters(),lr = 2e-4) for model in model.ebm_models]
    
    for epoch in range(9000):
        total_loss = 0
        for sample in tqdm(trainloader):
            # collect data from the sample
            images    = sample["image"];questions = sample["question"]
            programs  = sample["program"];answers = sample["answer"]
            # ground the concept in the image by neuro-symbolic programs
            results = model.ground_concept(images,programs)
            
            # ground the results from the quasi-symbolic executor
            logprobs  = ground_results(results,answers)

            # add all the terms from the quasi-symbolic executor to the qa-loss
            qa_loss = 0
            for term in logprobs:qa_loss -= term
            working_loss = 0
            
            if joint:
                
                latents = model.component_model.embed_latent(images)
                latents = torch.chunk(latents, ebm_config.components, dim=1)
                im_neg = torch.rand_like(images)
                im_neg += 0.1 * images.detach()

                im_neg, im_negs, im_grad, _ = gen_image(latents,model.ebm_models,images ,im_neg, 5)
                im_negs = torch.stack(im_negs, dim=1)

                energy_pos = 0
                energy_neg = 0

                energy_poss = []
                energy_negs = []
                for i in range(ebm_config.components):
                    energy_poss.append(model.ebm_models[i].forward(images, latents[i]))
                    energy_negs.append(model.ebm_models[i].forward(im_neg.detach(), latents[i]))

                energy_pos = torch.stack(energy_poss, dim=1)
                energy_neg = torch.stack(energy_negs, dim=1)
                ml_loss = torch.abs(energy_pos - energy_neg).mean()

                im_loss = torch.pow(im_negs[:, -1:] - images[:, None], 2).mean()
                working_loss += im_loss + 0.01 * ml_loss
                energy_loss = ml_loss;recons_loss = im_loss
                plt.figure("images")
                plt.subplot(1,2,1);plt.cla()
                plt.imshow(im_neg.cpu().detach()[0].permute([1,2,0]))
                plt.subplot(1,2,2);plt.cla()
                plt.imshow(images.cpu()[0].permute([1,2,0]))

                if isinstance(programs[0],str):programs= toFuncNode(programs[0])
                plt.figure("program")
                plt.cla()
                func_g = toNxGraph(programs)
                nx.draw_spectral(func_g,with_labels = True)
                plt.text(-0.6,0.4,questions[0])
                plt.pause(0.0001)
            # if it is a joint training, add the reconstruction loss and energy loss from the ebm

            # optimize the concept structure by reduce the the qa-loss Pr[a,e(c,im)]
            working_loss += qa_loss * 0.001
            working_loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            if joint:
                [torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0) for model in model.ebm_models]
                [optimizer.step() for optimizer in ebm_optimizers]
                [optimizer.zero_grad() for optimizer in ebm_optimizers]
            total_loss += working_loss
        loss_history.append(total_loss.detach())
        if joint:
            print("epoch: {} qa_loss:{} recons: {} energy_loss:{}".format(epoch,qa_loss,recons_loss,energy_loss))
        else:
            print("epoch: {} total_loss:{}".format(epoch,total_loss))
        plt.figure("namomo")
        torch.save(model,"checkpoints/model100.ckpt")
        plt.cla();plt.plot(loss_history);plt.pause(0.0001);
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    from config import *
    from machine_prototypes.ebm_learner import *

    # create the energy-based meta-concept learner
    sp3_concepts =\
        {"static_concepts" :nn.ModuleList([ConceptBox("red",  ctype = "color",dim = config.concept_dim),
                             ConceptBox("green",ctype = "color",dim = config.concept_dim),
                             ConceptBox("blue", ctype = "color",dim = config.concept_dim),
                             ConceptBox("cube",ctype = "category",dim = config.concept_dim),
                             ConceptBox("circle",ctype = "category",dim = config.concept_dim),
                             ConceptBox("diamond",ctype = "category",dim = config.concept_dim),]),
        "dynamic_concepts":[],
        "relations":[]}
    ebml = EBMLearner(config,sp3_concepts)
    ebml = torch.load("checkpoints/model100.ckpt")
    #ebml.component_model = torch.load("comet.ckpt")
    sprite3dataset = Sprite3("train")

    #train_comet(sprite3dataset,ebml.component_model,epoch = 500)
    train_ebml(ebml,sprite3dataset,joint = True)
