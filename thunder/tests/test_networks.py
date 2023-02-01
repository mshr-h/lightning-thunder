from functools import partial, reduce
from itertools import product
import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.testing import assert_close, make_tensor

import thunder
import thunder.core.dtypes as datatypes
import thunder.core.lang as tlang
import thunder.langs.torch as ttorch

from .framework import Executor, executors, NOTHING, nvFuser, requiresCUDA, TorchEx

# TODO: consider making a deterministic dropout or common source of randomness to enable the dropout
#   comparison
@executors(dtypes=(thunder.float32,))
def test_nanogpt_mlp_functional(executor, device, dtype):
    def new_gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

    n = 4

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.c_fc = nn.Linear(n, 4 * n)
            self.c_proj = nn.Linear(4 * n, n)
            self.dropout = nn.Dropout()

        def forward(self, x):
            x = self.c_fc(x)
            x = new_gelu(x)
            x = self.c_proj(x)
            # x = self.dropout(x)
            return x

    def thunder_new_gelu(x):
        return 0.5 * x * (1.0 + ttorch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * ttorch.pow(x, 3.0))))

    def thunder_forward_functional(a, c_fc_weight, c_fc_bias, c_proj_weight, c_proj_bias):
        b = ttorch.linear(a, c_fc_weight) + c_fc_bias
        c = thunder_new_gelu(b)
        d = ttorch.linear(c, c_proj_weight, c_proj_bias)
        # e = ttorch.dropout(d)
        return d

    tdtype = ttorch.torch_dtype(dtype)
    make = partial(make_tensor, dtype=tdtype, device=device)
    mlp = MLP().to(device, dtype=tdtype)
    a = make((n, n))

    thunder_fn = thunder.make_traced(thunder_forward_functional, executor=executor)
    thunder_result = thunder_fn(a, mlp.c_fc.weight, mlp.c_fc.bias, mlp.c_proj.weight, mlp.c_proj.bias)
    torch_result = mlp(a)

    assert_close(thunder_result, torch_result)
