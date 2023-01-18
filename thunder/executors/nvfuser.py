from enum import auto, Enum
from typing import Sequence
import time
from functools import partial

import torch
import torch._C._nvfuser as nvfuser
from torch._C._nvfuser import DataType, Fusion, FusionDefinition

from thunder.core.pytree import tree_flatten, tree_unflatten, tree_map
import thunder.core.dtypes as dtypes
import thunder.langs.torch as ttorch
from thunder.core import prims, utils
from thunder.core.proxies import Proxy, NumberProxy, TensorProxy

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
    dtypes.complex128: DataType.ComplexDouble,
    dtypes.complex64: DataType.ComplexFloat,
    dtypes.float64: DataType.Double,
    dtypes.float32: DataType.Float,
    dtypes.float16: DataType.Half,
    dtypes.bfloat16: DataType.BFloat16,
    dtypes.int64: DataType.Int,
    dtypes.int32: DataType.Int32,
    dtypes.bool8: DataType.Bool,
}

_thunder_dtype_to_nvfuser_dtype_scalar_map = {
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
    # prims.Ops.ACOSH: "acosh",
    prims.Ops.ASIN: "asin",
    prims.Ops.ATAN: "atan",
    prims.Ops.ATANH: "atanh",
    prims.Ops.BITWISE_NOT: "bitwise_not",
    prims.Ops.CEIL: "ceil",
    prims.Ops.COS: "cos",
    prims.Ops.COSH: "cosh",
    prims.Ops.ERF: "erf",
    prims.Ops.ERFC: "erfc",
    prims.Ops.EXP: "exp",
    prims.Ops.EXPM1: "expm1",
    prims.Ops.FLOOR: "floor",
    # The isfinite translation is incorrect, see https://github.com/csarofeen/pytorch/issues/2230
    # nvFuser's isfinite returns its output in the same datatype as the input,
    #   but prims.isfinite always expects a boolean return (consistent with
    #   Python, NumPy, JAX, and PyTorch)
    prims.Ops.ISFINITE: "isfinite",
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
    output_dtype = a.dtype
    if utils.is_complex_dtype(output_dtype):
        output_dtype = utils.corresponding_real_dtype(output_dtype)

    var = prims.reduction_meta(a, dim, output_dtype=output_dtype)
    mean = prims.reduction_meta(a, dim, output_dtype=a.dtype)

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
        not utils.is_complex_dtype(a),
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


# Creates an nvFuser input for the corresponding proxy
def _add_input(fd, x, proxy_to_nvfuser_map, used_inputs):
    # Constant case
    if not isinstance(x, Proxy):
        # Converts Tensor datatypes to torch datatypes
        if dtypes.is_dtype(x) and not dtypes.is_numbertype(x):
            return _thunder_dtype_to_nvfuser_dtype_map[x]
        return x

    nv = None
    if isinstance(x, NumberProxy):
        python_type = x.python_type
        nv_dtype = _thunder_dtype_to_nvfuser_dtype_scalar_map[python_type]
        nv = fd.define_scalar(nv_dtype)
    elif isinstance(x, TensorProxy):
        nv_dtype = _thunder_dtype_to_nvfuser_dtype_map[x.dtype]
        nv = fd.define_tensor(sizes=x.shape, strides=x.strides, dtype=nv_dtype)

    proxy_to_nvfuser_map[x] = nv
    used_inputs.append(x)

    return nv


# Finds or creates the nvFuser object associated with x,
#   possibly updating datastructures for proxies.
def _get_nv(x, *, fd, proxy_to_nvfuser_map, used_inputs):
    if x not in proxy_to_nvfuser_map:
        return _add_input(fd, x, proxy_to_nvfuser_map, used_inputs)

    return proxy_to_nvfuser_map[x]


# TODO: support NumPy arrays
# TODO: possibly support caching on the object that fusion returns
# fuse returns a function that, when called with actual PyTorch tensors and Python numbers
#   in place of the corresponding TensorProxies and NumberProxies, computes the given
#   trace.
# NOTE: the function can be reused, but it should be called with tensors that have the
#   same metadata, numbers of the same type, all conditionals on the number evaluated
#   the same as previous number inputs, and all other values constant.
def _fuse(trace):
    proxy_to_nvfuser_map = {}
    used_inputs = []
    outputs = []
    flat_outputs, output_structure = tree_flatten(trace.outputs)

    fs = Fusion()
    with FusionDefinition(fs) as fd:
        __get_nv = partial(_get_nv, fd=fd, proxy_to_nvfuser_map=proxy_to_nvfuser_map, used_inputs=used_inputs)

        #
        for sym in trace.symbols:
            nv_args = tree_map(__get_nv, sym.args)
            nv_kwargs = tree_map(__get_nv, sym.kwargs)
            nv_op = _get_nvfuser_op(fd, sym.op)
            nv_result = nv_op(*nv_args, **nv_kwargs)

            # Associates proxies to the nvFuser results
            # NOTE: it's assumed that NV operations produce results with proxies as leaves
            proxies, _ = tree_flatten(sym.result)
            nvs, _ = tree_flatten(nv_result)
            for p, nv in zip(proxies, nvs):
                assert p not in proxy_to_nvfuser_map
                assert isinstance(p, Proxy)
                proxy_to_nvfuser_map[p] = nv

        #

        nvfuser_output_ctr = 0
        for idx, o in enumerate(flat_outputs):
            if isinstance(o, Proxy):
                # NOTE: Not every output will be in this map, because the output
                #   will not have been produced by an nvFuser operator if
                #   it's also an input.
                if o in proxy_to_nvfuser_map:
                    fd.add_output(proxy_to_nvfuser_map[o])
                    outputs.append(nvfuser_output_ctr)
                    nvfuser_output_ctr += 1
                else:
                    outputs.append(o)

    # Builds the callable

    tab = "  "
    cstr = f"def fusion(args, kwargs):"

    # Acquires inputs
    flat_positional_inputs, _ = tree_flatten(trace.args)
    flat_kwarg_inputs, _ = tree_flatten(trace.kwargs)

    cstr += f"\n{tab}# Extracts inputs"
    cstr += f"\n{tab}flat_args, _ = tree_flatten(args)"
    cstr += f"\n{tab}flat_kwargs, _ = tree_flatten(kwargs)"
    for idx, pinp in enumerate(flat_positional_inputs):
        if isinstance(pinp, Proxy):
            cstr += f"\n{tab}{pinp.name} = flat_args[{idx}]"
    for idx, kwinp in enumerate(flat_kwarg_inputs):
        if isinstance(kwinp, Proxy):
            cstr += f"\n{tab}{kwinp.name} = flat_kwargs[{idx}]"

    # Calls fusion
    cstr += f"\n{tab}# Invokes fusion"

    arg_string = ", ".join(tuple(uinp.name for uinp in used_inputs))
    cstr += f"\n{tab}result = _fusion(({arg_string}))"

    # Assembles output
    output_strs = []
    for o in outputs:
        if isinstance(o, Proxy):
            output_strs.append(o.name)
        else:
            output_strs.append(f"result[{o}]")
    output_str = ", ".join(output_strs)

    cstr += f"\n{tab}# Assembles output"
    cstr += f"\n{tab}return tree_unflatten(({output_str}), output_structure)"

    # Creates context
    ctx = {
        "tree_flatten": tree_flatten,
        "tree_unflatten": tree_unflatten,
        "_fusion": fs.execute,
        "output_structure": output_structure,
    }

    print(cstr)

    code = compile(cstr, "nvfuser.gen", mode="exec")
    exec(code, ctx)
    fusion = ctx["fusion"]

    return fusion


class nvFuserCtx:
    def __init__(self):
        pass

    def intercept(self, op):
        """"""

        # TODO: update match to not be on strings
        if op == "torch.var_mean":
            return var_mean

        return None

    def fuse(self, trace):
        return _fuse(trace)
