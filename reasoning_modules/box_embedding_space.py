import torch
import torch.nn as nn

from melkor_knowledge import *

def cast_to_entities(features):return [EntityBox(features[i:i+1]) for i in range(features.shape[0])]

class QuasiExecutor(nn.Module):
    def __init__(self,concepts):
        super().__init__()
        self.static_concepts = concepts["static_concepts"]
        self.dynamic_concepts = concepts["dynamic_concepts"]
        self.relations = concepts["relations"]

    def get_concept_by_name(self,name):
        for k in self.static_concepts:
            if k.name == name:return k
        assert False

    def sample_concept(self,concept = "AND(red,circle)"):
        if isinstance(concept,str):concept = toFuncNode(concept)
        def parse_concept(node):
            if node.token == "AND":return make_joint_concept(parse_concept(node.children[0]),parse_concept(node.children[1]))
            elif(0):pass
            else:return self.get_concept__by_name(node.token)
        return parse_concept(concept)

    def forward(self,program,context):
        if isinstance(program,str):program = toFuncNode(program)
        """
        the general scheme of the context is that there is always such terms in the
        result diction that have key:
            objects: the object features of perception
            scores:  the probability of each feature represents an actual object
        """
        def execute_node(node):
            if node.token == "scene":return context
            if node.token == "filter":
                input_set = execute_node(node.children[0])
                concept_name = node.children[1].token
                filter_concept = self.get_concept_by_name(concept_name)
                filter_pdf = calculate_filter_log_pdf(input_set["features"],filter_concept)
                filter_pdf = torch.min(filter_pdf,input_set["scores"])
                return {"features":input_set["features"],"scores":filter_pdf}
            if node.token == "exist":
                input_set = execute_node(node.children[0])
                exist_prob = torch.max(input_set["scores"]).exp()
                output_distribution = torch.log(torch.stack([exist_prob,1-exist_prob]))
                return {"outputs":["True","False"],"scores":output_distribution}
            if node.token == "count":
                input_set = execute_node(node.children[0])
                return torch.sum(input_set["scores"].exp())
            if node.token == "relate":
                return context
            if node.token == "unique":
                return context
            if node.token == "query":
                return context
            return 0
        results = execute_node(program)
        return results

def make_grid(concept):
    return concept

def realize_concept(concept):
    # if the concept is a primitive box
    # if the concept is union of concepts
    # if the concept is intersection of boxes
    return 0

def make_joint_concept(c1,c2):
    # if possible, make the intersection of two concepts
    joint_edge = M(c1,c2)-m(c1,c2)
    lower_bound = m(c1,c2);upper_bound = lower_bound + joint_edge
    center = ( lower_bound + upper_bound )/2.0 # the center of the box
    edge   = upper_bound - center # the edge of the box
    return ConceptBox("{} and {}".format(c1.name,c2.name),"complex",center,edge)

def sample(concept_box):
    # choose a random point in the concept box
    upper_bound = BoxMax(concept_box)
    lower_bound = BoxMin(concept_box)
    return torch.rand(upper_bound.shape) * (upper_bound - lower_bound) + lower_bound


if __name__ == "__main__":
    blue = ConceptBox("blue","color",dim = 64)
    red  = ConceptBox("red" ,"color",dim = 64)

    e1   = EntityBox(torch.randn([1,64]))
    e2   = EntityBox(torch.randn([1,64]))

    context = {"features":[e1,e1],"scores":torch.tensor([-0.01,-0.1])}

    print("start the test of the concept executor")

    concepts = {"static_concepts":torch.nn.ModuleList([blue,red]),"dynamic_concepts":[],"relations":["relations"]}
    executor = QuasiExecutor(concepts)


    optimizer = torch.optim.Adam(executor.parameters(),lr = 2e-3)

    for epoch in range(900):

        results = executor("exist(filter(scene(),red))",context);
        loss = 0-results["scores"][0]
        #print("epoch: {} logprob:{}".format(epoch,loss))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("# a test of execution on exist")
    results = executor("exist(scene())",context)
    print(results["outputs"])
    print(results["scores"].exp())

    print("# a test of execution on filter")
    results = executor("exist(filter(scene(),red))",context)
    print(results["outputs"])
    print(results["scores"].exp())

    print("# a test of execution on count")
    results = executor("count(scene())",context)
    print(results)    
    
    complex = make_joint_concept(red,blue)
    print(complex.name)
    c3 = sample(red)
