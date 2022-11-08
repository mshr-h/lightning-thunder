from typing import Sequence

from thunder.core import prims
from thunder.core.proxies import TensorProxy, IntegerProxy

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
            else:
                raise AssertionError(f"execute(): Received unknown input type: {arg}")

        for sym in t.symbols:
            nv_op = _get_nvfuser_op(fd, sym.op)

            # TODO: support symbolic integer proxies
            def _proxy_to_nv(p):
                # TODO: always enumerating every element of a sequence seems expensive
                if isinstance(p, Sequence):
                    return tuple(map(_proxy_to_nv, p))
                if isinstance(p, IntegerProxy):
                    return p.value
                if isinstance(p, TensorProxy):
                    return proxy_to_nv_map[p.name]

                return p

            nv_args = tuple(map(_proxy_to_nv, sym.args))

            # TODO: support multiple returns
            nv_result = nv_op(*nv_args)
            proxy_to_nv_map[sym.result.name] = nv_result

        for out in t.outputs:
            fd.add_output(proxy_to_nv_map[out.name])

    # TODO: consider generalizing to other arg types
    nvf_out = fs.execute(args)[0]
    return nvf_out, fs
