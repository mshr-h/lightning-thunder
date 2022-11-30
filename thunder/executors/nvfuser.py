from enum import auto, Enum
from typing import Sequence

import torch
import torch._C._nvfuser as nvfuser
from torch._C._nvfuser import DataType, Fusion, FusionDefinition

import thunder.core.dtypes as dtypes
import thunder.langs.torch as ttorch
from thunder.core import prims, utils
from thunder.core.proxies import NumberProxy, TensorProxy

nvTensor = torch._C._nvfuser.Tensor

__all__ = [
    "nvfuser",
]


class nvOps(Enum):
    VAR_MEAN = auto()


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

_thunder_dtype_to_nvfuser_dtype_map = {
    dtypes.complex128_: DataType.ComplexDouble,
    dtypes.complex128: DataType.ComplexDouble,
    dtypes.complex64_: DataType.ComplexFloat,
    dtypes.complex64: DataType.ComplexFloat,
    dtypes.float64_: DataType.Double,
    dtypes.float64: DataType.Double,
    dtypes.float32_: DataType.Float,
    dtypes.float32: DataType.Float,
    dtypes.float16_: DataType.Half,
    dtypes.float16: DataType.Half,
    dtypes.bfloat16_: DataType.BFloat16,
    dtypes.bfloat16: DataType.BFloat16,
    dtypes.int64_: DataType.Int,
    dtypes.int64: DataType.Int,
    dtypes.int32_: DataType.Int32,
    dtypes.int32: DataType.Int32,
    # Python scalars
    complex: DataType.ComplexDouble,
    float: DataType.Double,
    int: DataType.Int,
    bool: DataType.Bool,
}


# Wrapper for prims.convert_element_type, necessary to convert dtype to nvfuser_dtype
def _convert_element_type_translation(fd):
    def _fn(a, dtype):
        # Handles Thunder's use of Python types as "weak" tensor dtypes
        # TODO: refactor into a helper
        if isinstance(a, nvTensor) and dtype in (bool, int, float, complex):
            tensor_dtype = torch.bool

            if dtype is int:
                tensor_dtype = dtypes.int64
            if dtype is float:
                tensor_dtype = dtypes.float32
            if dtype is complex:
                tensor_dtype = dtypes.complex64

            nvfuser_dtype = _thunder_dtype_to_nvfuser_dtype_map[tensor_dtype]
            return fd.ops.cast(a, nvfuser_dtype)

        nvfuser_dtype = _thunder_dtype_to_nvfuser_dtype_map[dtype]
        return fd.ops.cast(a, nvfuser_dtype)

    return _fn


# Maps the Thunder primitives to their corresponding nvfuser operation names
# TODO: map directly to the nvfuser operations, not their names
ops_to_nvfuser_ops_map = {
    # Data movement and transformation prims
    prims.Ops.CONVERT_ELEMENT_TYPE: _convert_element_type_translation,
    # Elementwise unary prims
    prims.Ops.ABS: "abs",
    prims.Ops.ACOS: "acos",
    # Elementwise binary prims
    prims.Ops.ADD: "add",
    prims.Ops.ATAN2: "atan2",
    prims.Ops.BITWISE_AND: "bitwise_and",
    prims.Ops.DIV: "div",
    prims.Ops.MUL: "mul",
    prims.Ops.SUB: "sub",
    # Shape prims
    prims.Ops.BROADCAST_IN_DIM: "broadcast_in_dim",
    # Reduction prims
    prims.Ops.SUM: "sum",
    prims.Ops.VAR: "var",
    nvOps.VAR_MEAN: "var_mean",
}


def _var_mean_prim_meta(a, dim, *, correction, **kwargs):
    output_dtype = a.thunder_dtype()
    if utils.is_complex_dtype(output_dtype):
        output_dtype = utils.corresponding_real_dtype(output_dtype)

    var = prims.reduction_meta(a, dim, output_dtype=output_dtype)
    mean = prims.reduction_meta(a, dim, output_dtype=a.thunder_dtype())

    return (var, mean)


var_mean_prim = prims.make_prim(nvOps.VAR_MEAN, "var_mean", _var_mean_prim_meta)


def var_mean(a, dim=None, unbiased=None, keepdim=False, *, correction=None):
    correction = ttorch._set_correction(unbiased, correction)

    # reduces over all dimensions if dim=() is passed
    if dim == () or dim == []:
        dim = None
    dim = ttorch._reduction_dims(a.shape, dim)

    # For complex tensors eager computes the variance as the sum of variances of
    # the real and imaginary parts
    # TODO: Creating a complex tensor from real and imaginary parts is not supported
    utils.check(
        not utils.is_complex_dtype(a.thunder_dtype()),
        lambda: "Complex tensors are not supported!",
    )

    v, m = var_mean_prim(a, dim, correction=correction)

    if keepdim:
        output_shape = [a.shape[i] if i not in dim else 1 for i in range(a.ndim)]
        broadcast_dims = [i for i in range(a.ndim) if i not in dim]
        v = prims.broadcast_in_dim(v, output_shape, broadcast_dims)
        m = prims.broadcast_in_dim(m, output_shape, broadcast_dims)

    return v, m


def _get_nvfuser_op(fd, op):
    nv_op = ops_to_nvfuser_ops_map[op]

    # TODO: always directly look up the appropriate callable
    if isinstance(nv_op, str):
        return getattr(fd.ops, ops_to_nvfuser_ops_map[op])

    # nv_op is a callable
    return nv_op(fd)


class nvFuserCtx:
    def __init__(self):
        pass

    def intercept(self, op):
        """"""

        # TODO: update match to not be on strings
        if op == "torch.var_mean":
            return var_mean

        return None

    def execute(self, *args, **kwargs):
        return execute(*args, **kwargs)


def _convert(fd, m, v, p):
    if isinstance(v, torch.Tensor):
        nv_dtype = _torch_dtype_to_nvfuser_dtype_map[v.dtype]
        nv = fd.define_tensor(sizes=v.shape, strides=v.stride(), dtype=nv_dtype)
        m[p.name] = nv
    elif isinstance(v, int):
        # NOTE: this handles both booleans and integers, since Python accepts bools as ints
        nv_dtype = _thunder_dtype_to_nvfuser_dtype_map[type(v)]
        nv = fd.define_scalar(nv_dtype)
        m[p.name] = nv
    elif isinstance(v, float):
        nv_dtype = _thunder_dtype_to_nvfuser_dtype_map[float]
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

            nv_result = nv_op(*nv_args, **nv_kwargs)

            # TODO handle more output datastructures
            if isinstance(nv_result, Sequence):
                for nvr, p in zip(nv_result, sym.result):
                    proxy_to_nv_map[p.name] = nvr
            else:
                proxy_to_nv_map[sym.result.name] = nv_result

        # TODO: test support for multiple return arguments
        for out in t.outputs:
            # TODO: handle more datastructures (probably want to tree map here)
            if isinstance(out, Sequence):
                for o in out:
                    fd.add_output(proxy_to_nv_map[o.name])
            else:
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

    nvf_out = fs.execute(args_and_kwargs)
    return nvf_out, fs
