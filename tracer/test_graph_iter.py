import torch

from colossalai.fx import ColoGraphModule

import sys
sys.path.append("/home/lczzh/ColoTracer/")
from tracer.colo_tracer_new import symbolic_trace

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        out = torch.add(x, 4)
        out = torch.sum(out)
        return out

model = MyModel()
gm = symbolic_trace(model, meta_args={"x":torch.rand((4, 4))})
for node in gm.graph.nodes:
    print(node.op, node.target)
    if node.target == torch.add:
        with gm.graph.inserting_after(node):
            new_node1 = gm.graph.call_function(torch.sum, node.args, node.kwargs)
            new_node2 = gm.graph.call_function(torch.abs, new_node1.args, new_node1.kwargs)
            node.replace_all_uses_with(new_node2)
