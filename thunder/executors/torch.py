from typing import Sequence
from collections import deque
from enum import Enum, auto
from functools import partial

from thunder.core import prims
from thunder.core import utils
from thunder.core.proxies import Proxy, NumberProxy, IntegerProxy, TensorProxy, dtypes

import thunder.langs.torch as ttorch

import torch

__all__ = [
    "torch",
]


def _convert_element_type_translation():
    def _fn(a, dtype):
        # Handles Thunder's use of Python types as "weak" tensor dtypes
        # TODO: refactor into a helper
        if isinstance(a, torch.Tensor) and dtype in (bool, int, float, complex):
            tensor_dtype = torch.bool

            if dtype is int:
                tensor_dtype = dtypes.int64
            if dtype is float:
                tensor_dtype = dtypes.float32
            if dtype is complex:
                tensor_dtype = dtypes.complex64

            torch_dtype = ttorch._thunder_to_torch_dtype_map[tensor_dtype]
            return a.to(torch_dtype)

        torch_dtype = ttorch._thunder_to_torch_dtype_map[dtype]
        return a.to(torch_dtype)

    return _fn


def _broadcast_in_dim_translation():
    def _fn(a, shape, broadcast_dims):
        s = list(shape)
        for broadcast_dim in broadcast_dims:
            s[broadcast_dim] = -1

        v = a
        for idx, x in enumerate(s):
            if x != -1:
                v = v.unsqueeze(idx)

        return v.expand(shape)

    return _fn


# Maps the Thunder primitives to their corresponding torch operation names
ops_to_torch_ops_map = {
    # Data movement and transformation prims
    prims.Ops.CONVERT_ELEMENT_TYPE: _convert_element_type_translation,
    # Elementwise unary prims
    prims.Ops.ABS: "abs",
    # Elementwise binary prims
    prims.Ops.ADD: "add",
    prims.Ops.ATAN2: "atan2",
    prims.Ops.BITWISE_AND: "bitwise_and",
    prims.Ops.DIV: "div",
    prims.Ops.SUB: "sub",
    # Shape prims
    prims.Ops.BROADCAST_IN_DIM: _broadcast_in_dim_translation,
    # Reduction prims
    prims.Ops.SUM: "sum",
    prims.Ops.VAR: "var",
}


def _get_torch_op(op):
    torch_op = ops_to_torch_ops_map[op]

    # TODO: always directly look up the appropriate callable
    if isinstance(torch_op, str):
        return getattr(torch, ops_to_torch_ops_map[op])

    # nv_op is a callable
    return torch_op()


class torchCtx(object):
    def __init__(self):
        pass

    def intercept(self, op):
        pass


def _convert(m, v, p):
    if isinstance(v, torch.Tensor):
        m[p.name] = v
    elif isinstance(v, int) or isinstance(v, float):
        # NOTE: this handles both booleans and integers, since Python accepts bools as ints
        m[p.name] = torch.tensor(v)
    else:
        # NOTE: we may extend this but we want to break when nvFuser breaks
        raise AssertionError(f"execute(): Received unknown input type: {v}")


# TODO: add kwarg support
def execute(t, *args, **kwargs):
    proxy_to_torch_map = {}

    # Converts inputs
    for arg, p in zip(args, t.inputs):
        _convert(proxy_to_torch_map, arg, p)

    for (k, v), (pk, pv) in zip(kwargs.items(), t.kwargs):
        _convert(proxy_to_torch_map, v, pv)

    # Convert constants
    for constant in t.constants:
        proxy_to_torch_map[constant.name] = constant.value

    for sym in t.symbols:
        torch_op = _get_torch_op(sym.op)

        def _proxy_to_value(x):
            if isinstance(x, NumberProxy):
                return x.value

            return x

        def _proxy_to_torch(x):
            # TODO: always enumerating every element of a sequence seems expensive
            #  (This comes up in calls to broadcast_in_dim where a list of IntegerProxies is passed as an argument)
            if isinstance(x, Sequence):
                return tuple(map(_proxy_to_torch, x))
            if isinstance(x, NumberProxy):
                # TODO: discuss with NVIDIA
                # NOTE: this means that symbols which are not inputs or constants are
                #   passed by value. Examples of these symbols are the lengths of the
                #   input tensors' dimensions.
                return proxy_to_torch_map.get(x.name, x.value)
            if isinstance(x, TensorProxy):
                return proxy_to_torch_map[x.name]

            return x

        torch_args = tuple(map(_proxy_to_torch, sym.args))
        torch_kwargs = {k: _proxy_to_torch(v) for k, v in sym.kwargs.items()}

        torch_result = torch_op(*torch_args, **torch_kwargs)

        # TODO handle more output datastructures
        if isinstance(torch_result, Sequence):
            for r, p in zip(torch_result, sym.result):
                proxy_to_torch_map[p.name] = r
        else:
            proxy_to_torch_map[sym.result.name] = torch_result

    torch_out = []

    # TODO: test support for multiple return arguments
    for out in t.outputs:
        # TODO: handle more datastructures (probably want to tree map here)
        if isinstance(out, Sequence):
            for o in out:
                torch_out.append(proxy_to_torch_map[o.name])
        else:
            torch_out.append(proxy_to_torch_map[out.name])

    # Filters sequences in args, which are currently treated as constants
    # TODO: revisit this modeling
    def _arg_filter(x):
        if isinstance(x, Sequence):
            return True

        return False

    filtered_args = tuple(arg for arg in args if not _arg_filter(arg))

    # Adds kwargs
    flattened_kwargs = []
    for k, v in kwargs.items():
        flattened_kwargs.append(v)

    args_and_kwargs = filtered_args + tuple(flattened_kwargs)

    # we need to execute eagerly
    # nvf_out = fs.execute(args_and_kwargs)
    return torch_out, None
