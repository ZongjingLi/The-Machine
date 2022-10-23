import torch
import torch.nn as nn

from melkor_knowledge import *

blue = ConceptBox("blue","color")
red  = ConceptBox("red" ,"color")

e1   = EntityBox(torch.randn([1,100]))
e2   = EntityBox(torch.randn([1,100]))

lp = logJointVolume(e1,blue,True)

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
            return 0
        results = execute_node(program,context)
        return results



print(lp)

print(calculate_categorical_log_pdf(e1,[blue,red]).exp())

print(calculate_filter_log_pdf([e1,e2],red))

