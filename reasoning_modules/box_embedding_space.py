import torch
import torch.nn as nn

from melkor_knowledge import *

blue = ConceptBox("blue","color")
red  = ConceptBox("red" ,"color")

e1   = EntityBox(torch.randn([1,100]))
e2   = EntityBox(torch.randn([1,100]))

context = {"objects":[e1,e1],"scores":torch.tensor([0.0,0.0])}

class QuasiExecutor(nn.Module):
    def __init__(self,concepts):
        super().__init__()
        self.static_concepts = concepts["static_concepts"]
        self.dynamic_concepts = concepts["dynamic_concepts"]
        self.relations = concepts["relations"]

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
            if node.token == "filter":return context
            if node.token == "exist":
                input_set = execute_node(node.children[0])
                exist_prob = torch.max(input_set["scores"]).exp()
                output_distribution = torch.log(torch.tensor([exist_prob,1-exist_prob]))
                return {"outputs":["True","False"],"scores":output_distribution}
            return 0
        results = execute_node(program)
        return results





if __name__ == "__main__":
    lp = logJointVolume(e1,blue,True)
    print(lp)

    print(calculate_categorical_log_pdf(e1,[blue,red]).exp())

    print(calculate_filter_log_pdf([e1,e2],red))

    print("start the test of the concept executor")

    concepts = {"static_concepts":[e1,e2],"dynamic_concepts":[],"relations":["relations"]}
    executor = QuasiExecutor(concepts)

    results = executor("exist(scene())",context)

    print(results["outputs"])
    print(results["scores"].exp())