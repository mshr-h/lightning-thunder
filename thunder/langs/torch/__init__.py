import operator
from enum import Enum
from functools import partial, reduce
from typing import Callable, Optional, Sequence, Tuple
from numbers import Number

import torch

import thunder.core.trace as trace
import thunder.core.utils as utils
from thunder.core.utils import langctx
import thunder.core.proxies as proxies
from thunder.core.proxies import TensorProxy
import thunder.core.dtypes as dtypes
import thunder.core.lang as tlang
import thunder.core.prims as prims
from thunder.core.proxies import TensorProxy

__all__ = [
    # Elementwise Unary Ops
    "acos",
    # Elementwise Binary Ops
    "add",
    "mul",
    # Reduction Ops
    "_set_correction",
    "_reduction_dims",
    "mean",
    "var",
    "var_mean",
    # Language context
    "TorchLangCtx",
]

# The Torch language

# TODO: language contexts like Torch could be
#   expanded to allow datatypes that the original language didn't support
_thunder_to_torch_dtype_map = {
    bool: torch.bool,
    int: torch.int32,
    float: torch.float32,
    complex: torch.complex64,
    dtypes.bool8_: torch.bool,
    dtypes.bool8: torch.bool,
    dtypes.uint8_: torch.uint8,
    dtypes.uint8: torch.uint8,
    dtypes.int8_: torch.int8,
    dtypes.int8: torch.int8,
    dtypes.int16_: torch.int16,
    dtypes.int16: torch.int16,
    dtypes.int32_: torch.int32,
    dtypes.int32: torch.int32,
    dtypes.int64_: torch.int64,
    dtypes.int64: torch.int64,
    dtypes.bfloat16_: torch.bfloat16,
    dtypes.bfloat16: torch.bfloat16,
    dtypes.float16_: torch.float16,
    dtypes.float16: torch.float16,
    dtypes.float32_: torch.float32,
    dtypes.float32: torch.float32,
    dtypes.float64_: torch.float64,
    dtypes.float64: torch.float64,
    dtypes.complex32_: torch.complex32,
    dtypes.complex32: torch.complex32,
    dtypes.complex64_: torch.complex64,
    dtypes.complex64: torch.complex64,
    dtypes.complex128_: torch.complex128,
    dtypes.complex128: torch.complex128,
}

_torch_to_thunder_dtype_map = {
    v: k
    for k, v in _thunder_to_torch_dtype_map.items()
    if not utils.is_weak_dtype(k) and not (type(k) is type and issubclass(k, Number))
}


def thunder_dtype(torch_dtype):
    return _torch_to_thunder_dtype_map[torch_dtype]


def torch_dtype(thunder_dtype):
    return _thunder_to_torch_dtype_map[thunder_dtype]


class TorchLangCtx(object):

    # NOTE: language context is a singleton
    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(TorchLangCtx, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        self.dtype_cls = torch.dtype
        self.tensor_cls = torch.Tensor

    def proxy(self, x, *, name):
        if isinstance(x, torch.Tensor):
            dtype = thunder_dtype(x.dtype)
            return TensorProxy(name=name, shape=x.shape, device=str(x.device.type), dtype=dtype)
        else:
            return proxies.proxy(x, name=name)

    def thunder_dtype(torch_dtype):
        return _torch_to_thunder_dtype_map[torch_dtype]

    def torch_dtype(thunder_dtype):
        return _thunder_to_torch_dtype_map[thunder_dtype]

    #
    # Tensor methods
    #

    #
    # Elementwise Unary Methods
    #
    def acos(self, a):
        return acos(a)

    #
    # Elementwise Binary Methods
    #

    # +
    def add(self, a, b):
        return tlang.add(a, b)

    # *
    def mul(self, a, b):
        return tlang.mul(a, b)

    # -
    def sub(self, a, b):
        return tlang.sub(a, b)

    # /
    def true_divide(self, a, b):
        return tlang.true_divide(a, b)

    #
    # Reduction Methods
    #

    def var(self, *args, **kwargs):
        return var(*args, **kwargs)


def ctx():
    return TorchLangCtx()


#
# Elementwise Unary Ops
#


def acos(a):
    return tlang.acos(a)


#
# Elementwise Binary Ops
#


def add(a, b, *, alpha=None):
    if alpha is not None:
        b = b * alpha

    return a + b


def mul(a, b):
    return a * b


#
# Reduction Ops
#


class REDUCTION_OUTPUT_TYPE_KIND(Enum):
    SAME = (0,)
    COMPLEX_TO_FLOAT = (1,)
    # Keeps the output in the computation type (used for mean)
    KEEP_PROMOTED_TYPE = (2,)
    ALWAYS_BOOL = (3,)


# Maps lower precision datatypes to their corresponding computation datatypes
_computation_dtype_map = {
    torch.bfloat16: torch.float32,
    torch.float16: torch.float32,
    torch.complex32: torch.complex64,
}


def get_computation_dtype(dtype: torch.dtype) -> torch.dtype:
    return _computation_dtype_map.get(dtype, dtype)


def _reduction_dtypes(
    arg,
    output_dtype_kind: REDUCTION_OUTPUT_TYPE_KIND,
    dtype=None,
):
    # even though some reductions, like amin or amax, don't strictly require type promotion,
    # all the math ops (including comparisons) are still defined only for a computation type,
    # so promotion will still happen. We are doing it explicitly here
    inp_dtype = dtype if dtype is not None else arg.dtype
    computation_dtype = get_computation_dtype(inp_dtype)
    if (
        output_dtype_kind == REDUCTION_OUTPUT_TYPE_KIND.SAME
        or output_dtype_kind == REDUCTION_OUTPUT_TYPE_KIND.COMPLEX_TO_FLOAT
    ):
        result_dtype = dtype if dtype else arg.dtype
        if output_dtype_kind == REDUCTION_OUTPUT_TYPE_KIND.COMPLEX_TO_FLOAT and utils.is_complex_dtype(result_dtype):
            result_dtype = utils.corresponding_real_dtype(result_dtype)
    elif output_dtype_kind == REDUCTION_OUTPUT_TYPE_KIND.KEEP_PROMOTED_TYPE:
        result_dtype = None
    else:  # ALWAYS_BOOL
        result_dtype = torch.bool
    return computation_dtype, result_dtype


def _reduction_dims(shape, dims: Optional[Sequence]) -> Tuple[int, ...]:
    if dims is None:
        return tuple(range(len(shape)))

    dims = tuple(utils.canonicalize_dim(len(shape), idx) for idx in dims)
    utils.check_no_duplicates(dims)

    return dims


# TODO: restore out support?
def _reduction(
    a,
    prim: Callable,
    *,
    has_identity: bool = True,
    accepts_dim_tuple: bool = True,  # to handle min/argmin that accept single dim only
    dims=None,
    keepdims: bool = False,
    dtype: Optional[torch.dtype] = None,  # should be specified for ops that support it
    output_dtype_kind: REDUCTION_OUTPUT_TYPE_KIND,
):
    # TODO: check that a is the correct type?

    utils.check(
        a.ndim <= 64,
        lambda: f"Received a tensor with {a.ndim} dimensions, but only tensors with up to 64 dims are supported!",
    )

    if not accepts_dim_tuple:
        assert dims is None or isinstance(dims, int)

    if isinstance(dims, int):
        dims = (dims,)

    dims = _reduction_dims(a.shape, dims)

    if not has_identity:
        valid_shape = (a.ndim == 0) or all(a.shape[i] for i in dims)
        utils.check(
            valid_shape,
            lambda: "Can't reduce over a zero-size dimension when computing a reduction without an identity value.",
        )

    computation_dtype, result_dtype = _reduction_dtypes(a, output_dtype_kind, dtype)

    a = tlang.maybe_convert_to_dtype(a, computation_dtype)
    result = prim(a, dims)

    if keepdims:
        output_shape = [a.shape[i] if i not in dims else 1 for i in range(a.ndim)]
        broadcast_dims = [i for i in range(a.ndim) if i not in dims]
        result = prims.broadcast_in_dim(result, output_shape, broadcast_dims)

    if result_dtype is not None:
        tlang.maybe_convert_to_dtype(result, result_dtype)

    return result


# Helper to handle the unbiased->correction deprecation on ops like var
def _set_correction(
    unbiased: Optional[bool] = None,
    correction: Optional[int] = None,
):
    if correction is not None and unbiased is not None:
        raise RuntimeError("cannot specify both correction and unbiased arguments")
    elif correction is None and unbiased is None:
        correction = 1
    elif correction is None and unbiased is not None:
        correction = 0 if unbiased is False else 1
    if not isinstance(correction, int):
        raise ValueError("correction argument should be integer")
    if correction < 0:
        raise ValueError("correction argument should be non-negative")
    return correction


def _dim_var_dispatch(dim=None, unbiased=None):
    # There's the following overload of torch.var:
    # var(Tensor self, bool unbiased=True) -> (Tensor, Tensor)
    # We need to explicitly convert bool dims to unbiased arg
    if unbiased is None and isinstance(dim, bool):
        unbiased = dim
        dim = None
    return dim, unbiased


def mean(a, dim=None, keepdim: bool = False, *, dtype=None):
    # reduces over all dimensions if dim=() is passed
    if dim == () or dim == []:
        dim = None
    if isinstance(dim, int):
        dim = (dim,)

    dtype = dtype if dtype is not None else a.dtype
    print(f"a={a}")
    print(f"dtype={dtype}")
    utils.check(
        not utils.is_integer_dtype(dtype) and not utils.is_boolean_dtype(dtype),
        lambda: "Datatype should be floating point or complex",
    )

    result = _reduction(
        a,
        prims.sum,
        dims=dim,
        keepdims=keepdim,
        dtype=dtype,
        output_dtype_kind=REDUCTION_OUTPUT_TYPE_KIND.KEEP_PROMOTED_TYPE,
    )

    dims = _reduction_dims(a.shape, dim)  # type: ignore[arg-type]
    nelem = 1 if a.ndim == 0 else reduce(operator.mul, (a.shape[i] for i in dims), 1)
    # TODO: the conversion of nelem to float won't be needed once type promotion is supported
    result = tlang.true_divide(result, float(nelem))
    result_dtype = a.dtype if dtype is None else dtype
    result = tlang.maybe_convert_to_dtype(result, result_dtype)
    return result


def var(
    a,
    dim=None,
    unbiased: Optional[bool] = None,
    keepdim: bool = False,
    *,
    correction: Optional[int] = None,
):
    dim, unbiased = _dim_var_dispatch(dim, unbiased)
    correction = _set_correction(unbiased, correction)
    # reduces over all dimensions if dim=() is passed
    if dim == () or dim == []:
        dim = None

    result = _reduction(
        a,
        partial(prims.var, correction=correction),
        dims=dim,
        keepdims=keepdim,
        dtype=None,
        has_identity=True,
        output_dtype_kind=REDUCTION_OUTPUT_TYPE_KIND.COMPLEX_TO_FLOAT,
    )
    return result


# TODO: consider being more aggressive about kwarg-only
# TODO: use of @langctx here is just for testing and could be removed
#  (the method call to var below would need to be replaced with a function call)
@langctx(ctx())
def var_mean(
    a,
    dim=None,
    unbiased: Optional[bool] = None,
    keepdim: bool = False,
    *,
    correction: Optional[int] = None,
):
    # TODO: programmatically add this redirection to all operations
    # TODO: avoid string construction
    intercepted = trace.get_executor_context().intercept("torch.var_mean")
    if intercepted is not None:
        return intercepted(a, dim, unbiased, keepdim, correction=correction)

    dim, unbiased = _dim_var_dispatch(dim, unbiased)
    v = a.var(dim, unbiased, keepdim, correction=correction)
    m = mean(a, dim, keepdim)
    return v, m
