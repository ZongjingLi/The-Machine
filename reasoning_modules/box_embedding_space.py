import torch
import torch.nn as nn

from melkor_knowledge import *

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
                output_distribution = torch.log(torch.tensor([exist_prob,1-exist_prob]))
                return {"outputs":["True","False"],"scores":output_distribution}
            if node.token == "count":
                input_set = execute_node(node.children[0])
                return torch.sum(input_set["scores"].exp())
            if node.token == "relate":
                return context
            return 0
        results = execute_node(program)
        return results





if __name__ == "__main__":
    blue = ConceptBox("blue","color")
    red  = ConceptBox("red" ,"color")

    e1   = EntityBox(torch.randn([1,100]))
    e2   = EntityBox(torch.randn([1,100]))

    context = {"features":[e1,e1],"scores":torch.tensor([-0.01,-0.1])}

    print(calculate_categorical_log_pdf(e1,[blue,red]).exp())

    print(calculate_filter_log_pdf([e1,e2],red))

    print("start the test of the concept executor")

    concepts = {"static_concepts":[blue,red],"dynamic_concepts":[],"relations":["relations"]}
    executor = QuasiExecutor(concepts)
    
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