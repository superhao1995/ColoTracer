import torch

from colossalai.fx import ColoGraphModule

import sys
sys.path.append("/home/lczzh/ColoTracer/")
from tracer.colo_tracer_new import symbolic_trace, ColoTracer

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(4, 4, bias=False)
        self.fc2 = torch.nn.Linear(4, 4, bias=False)

    def forward(self, x):
        out = self.fc1(x)
        out[0] = 0
        out = self.fc2(out)
        return out

model = MyModel()
meta_args = {'x': torch.rand(1, 4).to('meta')}
gm = symbolic_trace(model, meta_args=meta_args)
print(gm.code)
gm.graph.print_tabular()

data = torch.rand((1, 4))
non_fx_out = model(data)
fx_out = gm(data)
assert torch.allclose(fx_out, non_fx_out, atol=1e-5),\
    f'{model.__class__.__name__} has inconsistent outputs, {fx_out} vs {non_fx_out}'

