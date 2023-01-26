import pytest
import torch
from torch.testing import assert_close, make_tensor

import thunder
import thunder.core.lang as tlang
import thunder.core.prims as prims
import thunder.langs.torch as ttorch

from .framework import executors, JAX_AVAILABLE, NOTHING, requiresJAX

if JAX_AVAILABLE:
    import jax

import numpy as np


# TODO: extend OpInfos with a jax_reference option and refactor this test to be
#   auto-generated
# TODO: these test cases could be improved upon
@executors(dtypes=(thunder.float32,))
@requiresJAX
def test_broadcast_in_dim(executor, device, dtype):
    # The first 5 test cases below are taken from JAX's broadcast_in_dim tests
    #   https://github.com/google/jax/blob/main/tests/lax_test.py#L1171

    # inshape, outshape, dims
    cases = (
        ([2], [2, 2], [0]),
        ([2], [2, 2], [1]),
        ([2], [2, 3], [0]),
        ([], [2, 3], []),
        ([1], [2, 3], [1]),
        ((4, 6, 3, 1), (5, 4, 7, 6, 3, 6, 6), (1, 3, 4, 5)),
    )

    def foo(a, shape, broadcast_dims):
        return prims.broadcast_in_dim(a, shape, broadcast_dims)

    traced_bid = thunder.make_traced(foo, executor=executor)
    tdtype = ttorch.torch_dtype(dtype)

    for inshape, outshape, dims in cases:
        a = make_tensor(inshape, device=device, dtype=tdtype)
        thunder_result = traced_bid(a, outshape, dims)

        a_np = a.cpu().numpy()
        jax_result = jax.lax.broadcast_in_dim(a_np, outshape, dims)
        jax_result = torch.from_numpy(np.array(jax_result))

        assert_close(thunder_result.cpu(), jax_result)
