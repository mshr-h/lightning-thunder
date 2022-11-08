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
# TODO: map directly to the nvfuser operations, not their names
ops_to_nvfuser_ops_map = {
    # Elementwise unary prims
    prims.Ops.ABS: "abs",
    # Elementwise binary prims
    prims.Ops.ADD: "add",
    prims.Ops.SUB: "sub",
    # Shape prims
    prims.Ops.BROADCAST_IN_DIM: "broadcast_in_dim",
}


def _get_nvfuser_op(fd, op):
    return getattr(fd.ops, ops_to_nvfuser_ops_map[op])


_torch_dtype_to_nvfuser_dtype_map = {
    torch.cdouble: DataType.ComplexDouble,
    torch.cfloat: DataType.ComplexFloat,
    torch.double: DataType.Double,
    torch.float: DataType.Float,
    torch.half: DataType.Half,
    torch.bfloat16: DataType.BFloat16,
    torch.long: DataType.Int,
    torch.int: DataType.Int32,
    torch.bool: DataType.Bool,
    # Python scalars
    complex: DataType.ComplexDouble,
    float: DataType.Double,
    int: DataType.Int,
    bool: DataType.Bool,
}


# TODO: add kwarg support
def execute(t, *args, shape_args=None):
    # Constructs a fusion from a trace
    proxy_to_nv_map = {}
    fs = Fusion()
    with FusionDefinition(fs) as fd:
        # Converts inputs
        for arg, p in zip(args, t.inputs):
            if isinstance(arg, torch.Tensor):
                nv_dtype = _torch_dtype_to_nvfuser_dtype_map[arg.dtype]
                nv = fd.define_tensor(
                    sizes=arg.shape, strides=arg.stride(), dtype=nv_dtype
                )
                proxy_to_nv_map[p.name] = nv
            elif isinstance(arg, int):
                nv_dtype = _torch_dtype_to_nvfuser_dtype_map[int]
                nv = fd.define_scalar(nv_dtype)
                proxy_to_nv_map[p.name] = nv
            elif isinstance(arg, float):
                nv_dtype = _torch_dtype_to_nvfuser_dtype_map[float]
                nv = fd.define_scalar(nv_dtype)
                proxy_to_nv_map[p.name] = nv
            else:
                raise AssertionError(f"execute(): Received unknown input type: {arg}")

        # TODO: experimental, only here for discussion
        # Adds symbolic shape information as new args
        # NOTE: this happens after regular args because we want this inputs
        #   to be defined after all others
        # length_args = deque()
        # for arg, p in zip(args, t.inputs):
        #     if isinstance(arg, torch.Tensor):
        #         for l, lp in zip(arg.shape, p.shape):
        #             nv_dtype = _torch_dtype_to_nvfuser_dtype_map[int]
        #             nv = fd.define_scalar(nv_dtype)
        #             proxy_to_nv_map[lp.name] = nv
        #             length_args.append(l)

        # Convert constants
        for constant in t.constants:
            nv = fd.define_constant(constant.value)
            proxy_to_nv_map[constant.name] = nv

        for sym in t.symbols:
            nv_op = _get_nvfuser_op(fd, sym.op)

            def _proxy_to_value(x):
                if isinstance(x, NumberProxy):
                    # NOTE: experimental for symbolic shapes (see above)
                    # return proxy_to_nv_map[x.name]
                    return x.value

                return x

            def _proxy_to_nv(x):
                # TODO: always enumerating every element of a sequence seems expensive
                #  (This comes up in calls to broadcast_in_dim where a list of IntegerProxies is passed as an argument)
                if isinstance(x, Sequence):
                    return tuple(map(_proxy_to_value, x))
                if isinstance(x, Proxy):
                    return proxy_to_nv_map[x.name]

                return p

            nv_args = tuple(map(_proxy_to_nv, sym.args))

            # TODO: support multiple returns from a call
            nv_result = nv_op(*nv_args)
            proxy_to_nv_map[sym.result.name] = nv_result

        # TODO: test support for multiple return arguments
        for out in t.outputs:
            fd.add_output(proxy_to_nv_map[out.name])

    # NOTE: experimental for symbolic shapes (see above)
    # nvf_out = fs.execute(args + tuple(length_args))[0]
    nvf_out = fs.execute(args)[0]
    return nvf_out, fs
