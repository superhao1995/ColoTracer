import torch

import sys
sys.path.append("/home/lczzh/ColoTracer/")
from tracer.colo_tracer_new import symbolic_trace

from colossalai.fx.profiler.tensor import MetaTensor

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = torch.nn.RNN(4, 4, num_layers=1, nonlinearity="relu", bidirectional=True)

    def forward(self, x):
        out = self.rnn(x)
        return out

model = MyModel().cuda()
meta_args = {'x': torch.rand(2, 4, 4).to('meta')}
gm = symbolic_trace(model, meta_args=meta_args)
print(gm.code)
gm.graph.print_tabular()

data = torch.rand((2, 4, 4)).cuda()
data = MetaTensor(data)
non_fx_out = model(data)

# print(model.rnn)

# fx_out = gm(data)
# assert torch.allclose(fx_out, non_fx_out, atol=1e-5),\
#     f'{model.__class__.__name__} has inconsistent outputs, {fx_out} vs {non_fx_out}'

