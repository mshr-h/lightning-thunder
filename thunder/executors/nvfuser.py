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
    # Reduction prims
    prims.Ops.VAR: "var",
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


def _convert(fd, m, v, p):
    if isinstance(v, torch.Tensor):
        nv_dtype = _torch_dtype_to_nvfuser_dtype_map[v.dtype]
        nv = fd.define_tensor(sizes=v.shape, strides=v.stride(), dtype=nv_dtype)
        m[p.name] = nv
    elif isinstance(v, int):
        # NOTE: this handles both booleans and integers, since Python accepts bools as ints
        nv_dtype = _torch_dtype_to_nvfuser_dtype_map[type(v)]
        nv = fd.define_scalar(nv_dtype)
        m[p.name] = nv
    elif isinstance(v, float):
        nv_dtype = _torch_dtype_to_nvfuser_dtype_map[float]
        nv = fd.define_scalar(nv_dtype)
        m[p.name] = nv
    else:
        raise AssertionError(f"execute(): Received unknown input type: {v}")


# TODO: add kwarg support
def execute(t, *args, **kwargs):
    # Constructs a fusion from a trace
    proxy_to_nv_map = {}
    fs = Fusion()
    with FusionDefinition(fs) as fd:
        # Converts inputs
        for arg, p in zip(args, t.inputs):
            _convert(fd, proxy_to_nv_map, arg, p)

        for (k, v), (pk, pv) in zip(kwargs.items(), t.kwargs):
            _convert(fd, proxy_to_nv_map, v, pv)

        # TODO: experimental, only here for discussion
        # NOTE: enabling this requires changes elsewhere -- just for illustration purposes
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
                    return x.value

                return x

            def _proxy_to_nv(x):
                # TODO: always enumerating every element of a sequence seems expensive
                #  (This comes up in calls to broadcast_in_dim where a list of IntegerProxies is passed as an argument)
                if isinstance(x, Sequence):
                    return tuple(map(_proxy_to_nv, x))
                if isinstance(x, NumberProxy):
                    # TODO: discuss with NVIDIA
                    # NOTE: this means that symbols which are not inputs or constants are
                    #   passed by value. Examples of these symbols are the lengths of the
                    #   input tensors' dimensions.
                    return proxy_to_nv_map.get(x.name, x.value)
                if isinstance(x, TensorProxy):
                    return proxy_to_nv_map[x.name]

                return x

            nv_args = tuple(map(_proxy_to_nv, sym.args))
            nv_kwargs = {k: _proxy_to_nv(v) for k, v in sym.kwargs.items()}

            # TODO: support multiple returns from a call
            nv_result = nv_op(*nv_args, **nv_kwargs)
            proxy_to_nv_map[sym.result.name] = nv_result

        # TODO: test support for multiple return arguments
        for out in t.outputs:
            fd.add_output(proxy_to_nv_map[out.name])

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

    nvf_out = fs.execute(args_and_kwargs)[0]
    return nvf_out, fs
