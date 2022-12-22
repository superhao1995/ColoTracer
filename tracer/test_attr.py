from typing import Any, Callable, Dict, Optional, Union

import torch
import torch.nn as nn
from torch.fx.graph_module import GraphModule
from torch.fx import Node
from torch.utils._pytree import tree_map
from colossalai.fx.tracer._tracer_utils import compute_meta_data_for_functions_proxy, extract_meta, is_element_in_list
from colossalai.fx.tracer.bias_addition_patch import func_to_func_dict, method_to_func_dict, module_to_func_dict
from colossalai.fx.tracer.registry import (
    bias_addition_function,
    bias_addition_method,
    bias_addition_module,
    meta_patched_function,
    meta_patched_module,
)


# from colossalai.fx import symbolic_trace as trace_ori

import sys
sys.path.append("/home/lczzh/ColoTracer/")
from tracer.colo_tracer_new import symbolic_trace as trace_new
from tracer.colo_tracer_new import meta_prop_pass
from tracer.symbolic_trace_ori import symbolic_trace as trace_ori



class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 4, bias=False)
        self.fc2 = nn.Linear(4, 4, bias=False)
        self.fc3 = nn.Linear(4, 4, bias=False)

    def forward(self, x):
        out = self.fc1(x)
        out += x
        # a = torch.zeros(out.shape, device="meta")
        # out += a
        out = self.fc2(out)
        out += x
        out = self.fc3(out)
        out += x
        out = torch.sum(out)
        return out


model = MyModel()
in_data1 = torch.rand((1, 4), device="meta")
meta_args1 = {"x": in_data1}
gm1 = trace_ori(model, meta_args=meta_args1)
in_data2 = torch.rand((1, 4), device="meta")
meta_args2 = {"x": in_data2}
gm2 = trace_new(model, meta_args=meta_args2)


meta_prop_pass(gm2, model, meta_args2)
node_list2 = []
for node in gm2.graph.nodes:
    node_list2.append(node)
idx = 0
for node in gm1.graph.nodes:
    attrs = dir(node_list2[idx])
    for n1_att1 in dir(node):
        if n1_att1 not in attrs:
            print(n1_att1, getattr(node, n1_att1))

    print(node.op, node.name, node_list2[idx]._meta_data)
    if node.op == "output":
        print(hasattr(node, "_meta_data"), hasattr(node_list2[idx], "_meta_data"))

    idx += 1

print(len(node_list2), idx)

