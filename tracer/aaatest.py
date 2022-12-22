
import torch
import numpy as np
from typing import Optional

# a = np.array([2,3])
# print(type(a))
# print(a.__index__())

def handle(cx: Optional[torch.Tensor] = None):
    print(cx)

handle(torch.rand(4))
handle(None)
