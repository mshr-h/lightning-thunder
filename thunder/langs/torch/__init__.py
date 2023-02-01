import operator
from enum import Enum
from functools import partial, reduce
from numbers import Number
from typing import Callable, Optional, Sequence, Tuple

import torch

import thunder.core.dtypes as dtypes
import thunder.core.lang as tlang
import thunder.core.prims as prims
import thunder.core.proxies as proxies
import thunder.core.trace as trace
import thunder.core.utils as utils
from thunder.core.proxies import TensorProxy
from thunder.core.utils import langctx

__all__ = [
    # Language context
    "ctx",
    "TorchLangCtx",
    # Tensor creation operations
    "full",
    "full_like",
    "uniform",
    "zeros_like",
    # Shape ops
    "reshape",
    "transpose",
    # Elementwise Unary Ops
    "abs",
    "acos",
    "acosh",
    "asin",
    "atan",
    "atanh",
    "bitwise_not",
    "cos",
    "exp",
    "rsqrt",
    "sin",
    "tanh",
    # Elementwise Binary Ops
    "add",
    "lt",
    "mul",
    "pow",
    "sub",
    "true_divide",
    # Elementwise Ternary Ops
    "masked_fill",
    "where",
    # Reduction Ops
    "_set_correction",
    "_reduction_dims",
    "amax",
    "mean",
    "sum",
    "var",
    "var_mean",
    # NN Ops
    # TODO: move to torch.nn.functional
    "dropout",
    "softmax",
    # Norm Ops
    # Matmul Ops
    "linear",
]

# The Torch language

#
# Language context
#

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


def ctx():
    return TorchLangCtx()


class TorchLangCtx:

    # NOTE: language context is a singleton
    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super().__new__(cls)
        return cls.instance

    def __init__(self):
        self.dtype_cls = torch.dtype
        self.tensor_cls = torch.Tensor

    def __repr__(self):
        return f"[TorchLangCtx]"

    def proxy(self, x, *, name):
        if isinstance(x, torch.Tensor):
            dtype = thunder_dtype(x.dtype)
            return TensorProxy(name=name, shape=x.shape, device=str(x.device.type), dtype=dtype, strides=x.stride())
        else:
            return proxies.proxy(x, name=name)

    def thunder_dtype(self, torch_dtype):
        return _torch_to_thunder_dtype_map[torch_dtype]

    def torch_dtype(self, thunder_dtype):
        return _thunder_to_torch_dtype_map[thunder_dtype]

    #
    # Tensor methods
    #

    # Attribute accesses
    def size(self, a):
        return a.shape

    #
    # Elementwise Unary Methods
    #
    def abs(self, a):
        return tlang.abs(a)

    def acos(self, a):
        return tlang.acos(a)

    def acosh(self, a):
        return tlang.acosh(a)

    def asin(self, a):
        return tlang.asin(a)

    def atan(self, a):
        return tlang.atan(a)

    def atanh(self, a):
        return tlang.atanh(a)

    def bitwise_not(self, a):
        return tlang.bitwise_not(a)

    def cos(self, a):
        return tlang.cos(a)

    def exp(self, a):
        return tlang.exp(a)

    #
    # Elementwise Binary Methods
    #

    # +
    def add(self, a, b):
        return tlang.add(a, b)

    # <
    def lt(self, a, b):
        return tlang.lt(a, b)

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


#
# Tensor Creation Ops
#

# TODO: check these signatures
def full(shape, fill_value, *, device, dtype=None):
    return tlang.full(shape, fill_value, device=device, dtype=dtype)


def full_like(tensor, fill_value, *, device=None, dtype=None):
    return tlang.full_like(tensor, fill_value, device=device, dtype=dtype)


# TODO: based on uniform_, check if Torch now has a functional uniform
# NOTE: the uniform_ documentation suggests the interval is specified using "from" and "to",
#   but from is a reserved keyword in Python
def uniform(shape, minval=0.0, maxval=1.0, *, device, dtype):
    return tlang.uniform(shape, minval, maxval, device=device, dtype=dtype)


# TODO: maybe just make this a passthrough?
def zeros_like(tensor, *, device=None, dtype=None):
    return full_like(tensor, 0.0, device=device, dtype=dtype)


#
# Shape Ops
#


def reshape(a, shape):
    return tlang.reshape(a, shape)


def transpose(a, dim0, dim1):
    dim0, dim1 = utils.canonicalize_dims(a.ndim, (dim0, dim1))

    permutation = list(range(0, a.ndim))
    permutation[dim0] = dim1
    permutation[dim1] = dim0
    return tlang.transpose(a, permutation)


#
# Elementwise Unary Ops
#


def acos(a):
    return tlang.acos(a)


def cos(a):
    return tlang.cos(a)


def exp(a):
    return tlang.exp(a)


def rsqrt(a):
    return tlang.rsqrt(a)


def sin(a):
    return tlang.sin(a)


def tanh(a):
    return tlang.tanh(a)


#
# Elementwise Binary Ops
#


def add(a, b, *, alpha=None):
    if alpha is not None:
        b = b * alpha

    return a + b


def lt(a, b):
    return tlang.lt(a, b)


def mul(a, b):
    return tlang.mul(a, b)


def pow(a, b):
    return tlang.pow(a, b)


def sub(a, b):
    return tlang.sub(a, b)


def true_divide(a, b):
    return tlang.true_divide(a, b)


#
# Elementwise ternary prims
#

# NOTE: masked_fill is a strange wrapper around where, it probably exists only because of PyTorch's inplace pattern
# NOTE: PyTorch's masked fill requires value be a number or number tensor
# NOTE: PyTorch's masked fill is only defined as a tensor method that implicitly takes a as the first argument
# NOTE: PyTorch's masked_fill_ requires the dtype of a not change, so it checks that
#   value can be safely cast to a
# TODO: PyTorch's masked_fill always returns a contiguous tensor
# TODO: add number tensor support
def masked_fill(a, mask, value):
    if isinstance(value, TensorProxy):
        raise NotImplementedError

    result = where(mask, value, a)
    return result


def where(pred, a, b):
    return tlang.where(pred, a, b)


#
# Reduction Ops
#


class REDUCTION_OUTPUT_TYPE_KIND(Enum):
    SAME = (0,)
    COMPLEX_TO_FLOAT = (1,)
    # Keeps the output in the computation type (used for mean)
    KEEP_PROMOTED_TYPE = (2,)
    ALWAYS_BOOL = (3,)


def _reduction_dtypes(
    arg,
    output_dtype_kind: REDUCTION_OUTPUT_TYPE_KIND,
    dtype=None,
):
    # even though some reductions, like amin or amax, don't strictly require type promotion,
    # all the math ops (including comparisons) are still defined only for a computation type,
    # so promotion will still happen. We are doing it explicitly here
    inp_dtype = dtype if dtype is not None else arg.dtype
    computation_dtype = utils.get_computation_dtype(inp_dtype)
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

    # reduces over all dimensions if dim=() is passed
    if dims == () or dims == []:
        dims = None
    if isinstance(dims, int):
        dims = (dims,)

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
        result = tlang.maybe_convert_to_dtype(result, result_dtype)

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


def amax(a, dim, keepdim):
    return _reduction(
        a,
        prims.amax,
        dims=dim,
        keepdims=keepdim,
        dtype=None,
        has_identity=False,
        output_dtype_kind=REDUCTION_OUTPUT_TYPE_KIND.SAME,
    )


def mean(a, dim=None, keepdim: bool = False, *, dtype=None):
    dtype = dtype if dtype is not None else a.dtype
    utils.check(
        not utils.is_integer_dtype(dtype) and not utils.is_boolean_dtype(dtype),
        lambda: f"dtype={dtype} is not a floating point or complex dtype",
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
    result = result / nelem
    result_dtype = a.dtype if dtype is None else dtype
    result = tlang.maybe_convert_to_dtype(result, result_dtype)
    return result


def sum(a, dim=None, keepdim=False, *, dtype=None):
    # Promotes low precision exact dtypes to int64
    if dtype is None:
        if utils.is_exact_dtype(a.dtype):
            dtype = dtypes.int64
        else:
            dtype = a.dtype

    result = _reduction(
        a,
        prims.sum,
        dims=dim,
        keepdims=keepdim,
        dtype=dtype,
        output_dtype_kind=REDUCTION_OUTPUT_TYPE_KIND.SAME,
    )

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
    v = var(a, dim, unbiased, keepdim, correction=correction)
    m = mean(a, dim, keepdim)
    return v, m


#
# NN Ops
#
# def _uniform_helper(shape, low=0., high=1., *, dtype, device):
#     utils.validate_shape(shape)

#     assert isinstance(low, Number)
#     assert isinstance(high, Number)

#     return prims._uniform_helper(shape, low=low, high=high, dtype=dtype, device=device)


def _dropout_helper(self, val):
    """Helper function for all dropout-type operators. During training, some of the elements of the input tensor are
    randomly masked.

    Returns the masked tensor of the boolean values.
    """

    r = uniform(self.shape, 0.0, 1.0, dtype=dtypes.float32, device=self.device)
    result = r < val

    return result


# full torch signature is: a, p, training, inplace
def dropout(a, p=0.5):
    utils.check(
        p <= 1 and p >= 0,
        lambda: f"dropout probability has to be between 0 and 1, but got, {p}",
    )

    if p == 1:
        return zeros_like(a)

    if p == 0:
        return a

    scale = 1 / (1 - p)
    dropout_mask = _dropout_helper(a, 1 - p)

    return a * dropout_mask * scale


# CompositeImplicitAutograd - don't register decomp
def softmax(a, dim, dtype=None):
    result_dtype = dtype or a.dtype
    computation_dtype = utils.get_computation_dtype(result_dtype)
    a_ = tlang.maybe_convert_to_dtype(a, computation_dtype)

    if a.numel() == 0:
        a_exp = exp(a_)
    else:
        a_max = amax(a_, dim, keepdim=True)
        a_exp = exp(a_ - a_max)

    result = true_divide(a_exp, sum(a_exp, dim, keepdim=True))
    converted = tlang.maybe_convert_to_dtype(result, result_dtype)
    return converted


#
# Norm Ops
#


def _normalize(a, norm_dims, eps):
    """Computes mean and 1/std of a tensor along norm_dims. Used as a helper function for normalization layers.

    Args:
        a (Tensor): input tensor
        norm_dims (DimsType): dimensions to normalize over
        eps (float): epsilon for numerical stability

    Returns:
        out (Tensor): normalized tensor.
        mean (Tensor): mean of the tensor along norm_dims.
        rstd (Tensor): 1/std of the tensor along norm_dims.
    """
    norm_dims = utils.canonicalize_dims(a.ndim, norm_dims)
    computation_dtype = utils.get_computation_dtype(a.dtype)
    a_acc = tlang.maybe_convert_to_dtype(a, computation_dtype)
    biased_var, mean = var_mean(a_acc, dim=norm_dims, unbiased=False, keepdim=True)
    rstd = rsqrt(biased_var + eps)
    out = (a - mean) * rstd
    return out, mean, rstd


# TODO: likely want to refactor these normalizations
def native_layer_norm(a, normalized_shape, weight, bias, eps):
    # Validates inputs
    normalized_ndim = len(normalized_shape)
    utils.check(normalized_ndim >= 1, lambda: f"Expected normalized_shape={normalized_shape} to have length >= 1!")
    # NOTE: canonicalizes the container for comparison to a tuple since
    # (1, 2, 3) != [1, 2, 3]
    utils.check(
        weight is None or weight.shape == tuple(normalized_shape),
        lambda: f"Expected weight.shape={weight.shape} to be the same as normalized_shape={normalized_shape}!",
    )
    utils.check(
        bias is None or bias.shape == tuple(normalized_shape),
        lambda: f"Expected bias.shape={bias.shape} to be the same as normalized_shape={normalized_shape}!",
    )
    # TODO: review this check -- seems like it's combining too much?
    utils.check(
        a.ndim >= normalized_ndim and a.shape[(a.ndim - normalized_ndim) :] == tuple(normalized_shape),
        lambda: f"TODO native_layer_norm error",
    )

    axis = a.ndim - normalized_ndim
    reduction_dims = list(range(axis, a.ndim))
    out, mean, rstd = _normalize(a, reduction_dims, eps)

    # Handles weight and bias
    if weight is not None:
        out = out * weight
    if bias is not None:
        out = out + bias

    out = tlang.maybe_convert_to_dtype(out, a.dtype)
    # TODO: review why this conversion cpu only?
    # if input.device.type == "cpu":
    mean = tlang.maybe_convert_to_dtype(mean, a.dtype)
    rstd = tlang.maybe_convert_to_dtype(rstd, a.dtype)

    return out, mean, rstd


def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    return native_layer_norm(input, normalized_shape, weight, bias, eps)[0]


#
# Matmul Ops
#


def linear(a, w, bias=None):
    return prims.linear(a, w, bias)


#
# torch -> thunder object mapping
#

_torch_to_thunder_function_map = {
    torch.add: add,
    torch.nn.functional.linear: linear,
}

_torch_to_thunder_complete_map = {
    **_torch_to_thunder_dtype_map,
    **_torch_to_thunder_function_map,
}
