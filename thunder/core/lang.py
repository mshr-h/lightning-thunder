from functools import reduce
from numbers import Number
from typing import Sequence

import thunder.core.dtypes as dtypes

# TODO: remove prims import
from . import prims, utils
from .proxies import NumberProxy, TensorProxy

# This file defines Thunder's "core" language.
#
# These operators are intended to be used when defining user-facing languages, like the torch or NumPy
# languages.
#
# This file depends on all other files in core.

__all__ = [
    # Data movement and transformation operations
    "maybe_convert_to_dtype",
    # Tensor creation operations
    "full",
    "full_like",
    "uniform",
    # Shape operations
    "expand",
    "reshape",
    # Elemenwise unary operations
    "abs",
    "acos",
    "acosh",
    "asin",
    "atan",
    "atanh",
    "bitwise_not",
    "ceil",
    "cos",
    "cosh",
    "erf",
    "erfc",
    "exp",
    "expm1",
    "floor",
    "isfinite",
    "rsqrt",
    "tanh",
    # Elementwise binary operations
    "add",
    "atan2",
    "bitwise_and",
    "lt",
    "mul",
    "pow",
    "sub",
    "true_divide",
    # Language context
    "CoreLangCtx",
]

#
# Data movement and transformation operations
#


# TODO: implement ref.cast with an option to enforce safe casting
def maybe_convert_to_dtype(a, dtype):
    """If a has the same dtype as the given dtype, returns a unmodified.

    Otherwise returns a converted to the given dtype.
    """

    utils.check(utils.is_dtype(dtype), lambda: f"Unknown dtype {dtype}!")

    if isinstance(a, Sequence):
        return tuple(maybe_convert_to_dtype(x, dtype) for x in a)
    if isinstance(a, TensorProxy):
        pass
    elif isinstance(a, Number):
        # NOTE: this allows conversions like (5, float32) -> 5., which is a little odd
        dtype = utils.dtype_to_numbertype(dtype)
    else:
        raise ValueError(
            f"Trying to convert the type of the data of an unknown object {a} of {type(a)} that is neither a tensor, number, or sequence!"
        )

    if not utils.are_same_dtypes(a, dtype):
        return prims.convert_element_type(a, dtype)

    return a


#
# Tensor creation operations
#

# TODO: add error checking
def full(shape, fill_value, *, device, dtype=None):
    fill_value_dtype = dtypes.to_dtype(fill_value)
    dtype = dtype if dtype is not None else fill_value_dtype

    # Ensures the requested fill_value can be safely cast to the dtype
    # NOTE: this is always true if the dtype is inferred
    utils.check(
        utils.can_safe_cast_number_to(fill_value, fill_value_dtype),
        lambda: f"Can't safely cast fill_value of numbertype {fill_value_dtype} to dtype {dtype}!",
    )

    return prims.full(shape, fill_value, device=device, dtype=dtype)


def full_like(tensor, fill_value, *, device=None, dtype=None):
    device = device if device is not None else tensor.device
    dtype = dtype if dtype is not None else tensor.true_dtype

    return full(tensor.shape, fill_value, device=device, dtype=dtype)


def uniform(shape, minval=0.0, maxval=1.0, *, dtype, device):
    return prims.uniform(shape, minval, maxval, dtype=dtype, device=device)


#
# Shape operations
#


def expand(a, *shape):
    # NOTE: cannot use utils.extract_shape_from_varargs here
    # because that also validates the shape, but the shape
    # given to expand may be "invalid"
    if len(shape) == 1 and isinstance(shape[0], Sequence):
        shape = tuple(shape[0])

    # TODO: improve this error message with error context
    utils.check(
        len(shape) >= len(a.shape),
        lambda: "expand: the requested shape has too few dimensions!",
    )

    offset = len(shape) - len(a.shape)
    shape_ = list(shape)
    for idx, x in enumerate(a.shape):
        offset_idx = idx + offset
        requested_length = shape[offset_idx]
        utils.check(
            requested_length == x or x == 1 or requested_length == -1,
            lambda: f"expand: attempting to expand a dimension of length {x}!",
        )

        shape_[offset_idx] = requested_length if requested_length != -1 else x

    # At this point shape must be valid
    # utils.check_valid_shape(shape_)

    return prims.broadcast_in_dim(a, shape_, tuple(range(offset, len(a.shape) + offset)))


# NOTE: shape may have a single -1 value, which is a marker that the length of that dimension
#   should be inferred
def reshape(a, shape):
    # Checks for -1 marker value
    numel = 1
    neg_one_idx = None
    for idx, l in enumerate(shape):
        if l >= 0:
            numel *= l
        else:
            utils.check(l == -1, "Found a negative dimension length {l} in shape={shape}!")
            utils.check(neg_one_idx is None, "Found two -1 markers in shape={shape}!")
            neg_one_idx = idx

    # Short-circuits if no shape inference is needed
    if neg_one_idx is None:
        return prims.reshape(a, shape)

    # Constructs the inferred shape, replacing -1 with the necessary length
    # TODO: this error message could probably be improved
    utils.check(a.numel() % numel == 0, "Can't infer length of dimension {neg_one_idx}!")
    remaining = a.numel() // numel
    shape = list(shape)
    shape[neg_one_idx] = remaining
    # NOTE: alternatively a new tuple could be constructed as follows:
    # shape = shape[:neg_one_idx] + (remaining,) + shape[neg_one_idx + 1:]
    return prims.reshape(a, shape)


def _compute_broadcast_shape(*_shapes):
    """Computes the common shape with the fewest dimensions that all input shapes can be broadcast to."""
    shapes = tuple(x for x in filter(lambda x: x is not None, _shapes))

    # Short-circuits if there are no inputs shapes
    #   This might happen in calls like add(2, 3)
    if len(shapes) == 0:
        return None

    common_shape = [
        1,
    ] * reduce(max, (len(shape) for shape in shapes))

    for shape in shapes:
        for idx in range(-1, -1 - len(shape), -1):
            if common_shape[idx] == 1:
                common_shape[idx] = shape[idx]

            utils.check(
                (shape[idx] == 1) or (common_shape[idx] == shape[idx]),
                lambda: f"Attempting to broadcast a dimension of length {shape[idx]}!",
            )

    return tuple(common_shape)


# TODO: add scalar support
# TODO: review hasattr pattern
def _maybe_broadcast(*args):
    """Returns tensors with the same shape, possibly broadcasting inputs to the result shape."""

    # Computes common shape
    common_shape = _compute_broadcast_shape(*map(lambda t: t.shape if hasattr(t, "shape") else None, args))

    def __maybe_broadcast(x, shape):
        if hasattr(x, "shape"):
            if not utils.same_shape(x.shape, common_shape):
                return expand(x, common_shape)

        return x

    return tuple(__maybe_broadcast(x, common_shape) for x in args)


#
# Elementwise unary operations
#
def _elementwise_unary_helper(prim, type_promotion_kind, a, *, supported_dtypes=None):
    computation_dtype, result_dtype = utils.elementwise_type_promotion(a, type_promotion_kind=type_promotion_kind)
    if supported_dtypes is not None:
        utils.check(
            computation_dtype in supported_dtypes,
            lambda: f"Unsupported dtype {computation_dtype}!",
        )

    a = maybe_convert_to_dtype(a, computation_dtype)

    result = prim(a)
    result = maybe_convert_to_dtype(result, result_dtype)

    return result


def abs(a):
    return _elementwise_unary_helper(prims.abs, utils.ELEMENTWISE_TYPE_PROMOTION_KIND.COMPLEX_TO_FLOAT, a)


def acos(a):
    return _elementwise_unary_helper(prims.acos, utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT, a)


def acosh(a):
    return _elementwise_unary_helper(prims.acosh, utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT, a)


def asin(a):
    return _elementwise_unary_helper(prims.asin, utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT, a)


def atan(a):
    return _elementwise_unary_helper(prims.atan, utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT, a)


def atanh(a):
    return _elementwise_unary_helper(prims.atanh, utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT, a)


def bitwise_not(a):
    return _elementwise_unary_helper(prims.bitwise_not, utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT, a)


def ceil(a):
    if utils.is_exact_dtype(utils.to_dtype(a)):
        return a

    return _elementwise_unary_helper(prims.ceil, utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT, a)


def cos(a):
    return _elementwise_unary_helper(prims.cos, utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT, a)


def cosh(a):
    return _elementwise_unary_helper(prims.cosh, utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT, a)


def erf(a):
    return _elementwise_unary_helper(prims.erf, utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT, a)


def erfc(a):
    return _elementwise_unary_helper(prims.erfc, utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT, a)


def exp(a):
    return _elementwise_unary_helper(prims.exp, utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT, a)


def expm1(a):
    return _elementwise_unary_helper(prims.expm1, utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT, a)


def floor(a):
    if utils.is_exact_dtype(utils.to_dtype(a)):
        return a

    return _elementwise_unary_helper(prims.floor, utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT, a)


def isfinite(a):
    if utils.is_exact_dtype(utils.to_dtype(a)):
        return full_like(a, True, dtype=dtypes.bool8)

    return _elementwise_unary_helper(prims.isfinite, utils.ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL, a)


def rsqrt(a):
    return _elementwise_unary_helper(prims.rsqrt, utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT, a)


def tanh(a):
    return _elementwise_unary_helper(prims.tanh, utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT, a)


#
# Elementwise binary operations
#

# Helper function that implements broadcasting and type promotion for elementwise binary operations
# TODO: consider making type promotion kind an annotation on operations so it can be queried
#   programmatically
def _elementwise_binary_helper(prim, type_promotion_kind, a, b, *, supported_dtypes=None):
    computation_dtype, result_dtype = utils.elementwise_type_promotion(a, b, type_promotion_kind=type_promotion_kind)

    a, b = _maybe_broadcast(a, b)

    if supported_dtypes is not None:
        utils.check(
            computation_dtype in dtypes.resolve_dtypes(supported_dtypes),
            lambda: f"Unsupported dtype {computation_dtype}!",
        )

    a, b = maybe_convert_to_dtype(a, computation_dtype), maybe_convert_to_dtype(b, computation_dtype)

    result = prim(a, b)
    result = maybe_convert_to_dtype(result, result_dtype)

    return result


def add(a, b):
    return _elementwise_binary_helper(prims.add, utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT, a, b)


def atan2(a, b):
    return _elementwise_binary_helper(prims.atan2, utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT, a, b)


def bitwise_and(a, b):
    return _elementwise_binary_helper(
        prims.bitwise_and,
        utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
        a,
        b,
        supported_dtypes=(dtypes.exact,),
    )


def lt(a, b):
    return _elementwise_binary_helper(prims.lt, utils.ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL, a, b)


def mul(a, b):
    return _elementwise_binary_helper(prims.mul, utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT, a, b)


def pow(a, b):
    return _elementwise_binary_helper(prims.pow, utils.ELEMENTWISE_TYPE_PROMOTION_KIND.BOOL_TO_LONG, a, b)


def sub(a, b):
    return _elementwise_binary_helper(prims.sub, utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT, a, b)


def true_divide(a, b):
    return _elementwise_binary_helper(prims.div, utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT, a, b)


class CoreLangCtx:
    def __init__(self):
        pass

    def add(self, a, b):
        return add(a, b)

    def sub(self, a, b):
        return sub(a, b)

    def true_divide(self, a, b):
        return true_divide(a, b)

    def intercept(self, op, *args, **kwargs):
        return None
