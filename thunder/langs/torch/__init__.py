from enum import Enum
from typing import Callable, Optional, Union, Sequence, Tuple
from functools import partial, reduce
import operator

import thunder.core.trace as trace
import thunder.core.utils as utils
from thunder.core.proxies import TensorProxy
import thunder.core.lang as tlang
import thunder.core.prims as prims

import torch

__all__ = [
    # Reduction Ops
    "_set_correction",
    "_reduction_dims",
    "mean",
    "var",
    "var_mean",
]

# The Torch language

# TODO: probably switch to just TensorProxy
TensorLike = (TensorProxy, torch.Tensor)
TensorLikeType = Union[TensorProxy, torch.Tensor]

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
    dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.dtype, Optional[torch.dtype]]:
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
        if (
            output_dtype_kind == REDUCTION_OUTPUT_TYPE_KIND.COMPLEX_TO_FLOAT
            and utils.is_complex_dtype(result_dtype)
        ):
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
    a: TensorLikeType,
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
            lambda: f"Can't reduce over a zero-size dimension when computing a reduction without an identity value.",
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
    utils.check(
        not utils.is_integer_dtype(dtype) and not utils.is_boolean_dtype(dtype),
        lambda: f"Dtype should be floating point or complex",
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
    v = var(a, dim, unbiased, keepdim, correction=correction)
    m = mean(a, dim, keepdim)
    return v, m
