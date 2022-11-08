from typing import Sequence
from collections import deque

from thunder.core import prims
from thunder.core.proxies import Proxy, NumberProxy, IntegerProxy, TensorProxy

import torch

from torch._C._nvfuser import (
    DataType,
    Fusion,
    FusionDefinition,
)

__all__ = [
    "nvfuser",
]

# Maps the Thunder primitives to their corresponding nvfuser operation names
# TODO: directly to the nvfuser operations, not their names
ops_to_nvfuser_ops_map = {
    prims.Ops.ADD: "add",
    prims.Ops.BROADCAST_IN_DIM: "broadcast_in_dim",
}


def _get_nvfuser_op(fd, op):
    return getattr(fd.ops, ops_to_nvfuser_ops_map[op])


# TODO: add constant support
# TODO: add support for non-tensor args
# TODO: add kwarg support
# TODO: review call conventions, tensor instantiation options and cache with NVIDIA
def execute(trace_or_fusion, *args):
    # Executes an existing fusion
    if isinstance(trace_or_fusion, Fusion):
        fs = trace_or_fusion
        nvf_out = fs.execute(args)[0]
        return nvf_out, fs

    # Constructs a fusion from a trace
    t = trace_or_fusion
    proxy_to_nv_map = {}
    fs = Fusion()
    with FusionDefinition(fs) as fd:
        assert len(args) == len(t.inputs)

        # Converts inputs
        for arg, p in zip(args, t.inputs):
            if isinstance(arg, torch.Tensor):
                nv = fd.define_tensor(sizes=arg.shape, strides=arg.stride())
                proxy_to_nv_map[p.name] = nv
            elif isinstance(arg, int):
                nv = fd.define_scalar(DataType.Int)
                proxy_to_nv_map[p.name] = nv
            else:
                raise AssertionError(f"execute(): Received unknown input type: {arg}")

        # Convert constants
        for constant in t.constants:
            nv = fd.define_constant(constant.value)
            proxy_to_nv_map[constant.name] = nv

        for sym in t.symbols:
            nv_op = _get_nvfuser_op(fd, sym.op)

            def _proxy_to_value(x):
                if isinstance(x, NumberProxy):
                    return x.value

                return x

            # TODO: support symbolic integer proxies
            def _proxy_to_nv(x):
                # TODO: always enumerating every element of a sequence seems expensive
                #  (This comes up in calls to broadcast_in_dim where a list of IntegerProxies is passed as an argument)
                if isinstance(x, Sequence):
                    return tuple(map(_proxy_to_value, x))
                if isinstance(x, Proxy):
                    return proxy_to_nv_map[x.name]

                return p

            nv_args = tuple(map(_proxy_to_nv, sym.args))

            # TODO: support multiple returns
            nv_result = nv_op(*nv_args)
            proxy_to_nv_map[sym.result.name] = nv_result

        # TODO: test support for multiple return arguments
        for out in t.outputs:
            fd.add_output(proxy_to_nv_map[out.name])

    nvf_out = fs.execute(args)[0]
    return nvf_out, fs
