from enum import Enum
from itertools import product
from functools import wraps
from numbers import Number
from typing import Callable, Sequence, Type

import thunder.core.trace as trace
import thunder.core.dtypes as datatypes
from .proxies import TensorProxy, NumberProxy


# This file defines utilities that can be used when defining primitive operations.

# This file depends on proxies.py.

__all__ = [
    # Error checking helpers
    "check",
    # Datatype and Python type-related functions
    "is_boolean_dtype",
    "is_integer_dtype",
    "is_low_precision_dtype",
    "is_float_dtype",
    "is_complex_dtype",
    "is_exact_dtype",
    "is_inexact_dtype",
    "is_number_type",
    "is_weak_dtype",
    "can_safe_cast_to",
    "corresponding_real_dtype",
    "corresponding_complex_dtype",
    "corresponding_weak_dtype",
    "corresponding_strong_dtype",
    "dtype_to_type",
    "check_same_dtype",
    "get_numberlike_type",
    "get_numberlike_value",
    "ELEMENTWISE_TYPE_PROMOTION_KIND",
    "elementwise_type_promotion",
    # Shape-related functions
    "is_number_tensor",
    "same_shape",
    "canonicalize_dim",
    "check_valid_length",
    "check_valid_shape",
    "check_no_duplicates",
]

#
# Error checking helpers
#


# TODO: maybe put check in a dependency-free base_utils that's also imported here (so it can be used by proxies.py)
def check(cond: bool, s: Callable[[], str], exception_type: Type[Exception] = RuntimeError) -> None:
    """Helper function for raising an error_type (default: RuntimeError) if a boolean condition fails.

    s is a callable producing a string to avoid string construction if the error check is passed.
    """
    if not cond:
        raise exception_type(s())


#
# Datatype-related functions
#


def _extract_dtype(x):
    if isinstance(x, datatypes.datatype):
        return x

    if isinstance(x, TensorProxy):
        return x.dtype

    if isinstance(x, Number):
        return get_numberlike_type(x)

    if x in (bool, int, float, complex):
        return x

    raise AssertionError(f"Trying to extract dtype from object {x} of unknown type {type(x)}!")


def is_boolean_dtype(dtype_) -> bool:
    dtype = _extract_dtype(dtype_)
    return dtype in (bool, datatypes.bool8, datatypes.bool8_)


def is_unsigned_dtype(dtype_):
    dtype = _extract_dtype(dtype_)
    return dtype in (bool, datatypes.bool8, datatypes.bool8_, datatypes.uint8_, datatypes.uint8)


def is_signed_integer_dtype(dtype_):
    dtype = _extract_dtype(dtype_)
    return is_integer_dtype(dtype) and not is_unsigned_dtype(dtype)


def is_integer_dtype(dtype_) -> bool:
    dtype = _extract_dtype(dtype_)
    return dtype in datatypes.integer_dtypes or dtype in (bool, int)


is_exact_dtype = is_integer_dtype


def is_low_precision_dtype(dtype_) -> bool:
    dtype = _extract_dtype(dtype_)
    return dtype in datatypes.low_precision_dtypes


def is_float_dtype(dtype_) -> bool:
    dtype = _extract_dtype(dtype_)
    return dtype in datatypes.float_dtypes or dtype is float


def is_complex_dtype(dtype_) -> bool:
    dtype = _extract_dtype(dtype_)
    return dtype in datatypes.complex_dtypes or dtype is complex


def is_inexact_dtype(dtype_):
    dtype = _extract_dtype(dtype_)
    return is_float_dtype(dtype) or is_complex_dtype(dtype)


def is_number_type(typ):
    return typ in (bool, int, float, complex)


def is_weak_dtype(dtype):
    if isinstance(dtype, datatypes.datatype):
        return dtype.is_weak

    return issubclass(dtype, Number)


def higher_dtype(a, b):
    for fn in (
        is_complex_dtype,
        is_float_dtype,
        is_signed_integer_dtype,
        is_unsigned_dtype,
        is_boolean_dtype,
    ):
        if fn(a):
            return a
        if fn(b):
            return b

    raise ValueError(f"Unknown inputs {a} and {b}!")


def can_safe_cast_to(*, cast_to, cast_from) -> bool:
    return higher_dtype(cast_to, cast_from) == cast_to


corresponding_real_dtype = datatypes.corresponding_real_dtype

corresponding_complex_dtype = datatypes.corresponding_complex_dtype

corresponding_weak_dtype = datatypes.corresponding_weak_dtype

corresponding_strong_dtype = datatypes.corresponding_strong_dtype


def dtype_to_type(dtype):
    """Computes the corresponding Python type (AKA "type kind") for the given dtype."""
    if is_boolean_dtype(dtype):
        return bool
    if is_integer_dtype(dtype):
        return int
    if is_float_dtype(dtype):
        return float
    if is_complex_dtype(dtype):
        return complex

    check(False, lambda: f"Unknown dtype {dtype}!")


def type_to_dtype(typ):
    if isinstance(typ, datatypes.datatype):
        return typ

    if typ is bool:
        return datatypes.bool8
    if typ is int:
        return datatypes.int64_
    if typ is float:
        return datatypes.float32_
    if typ is complex:
        return datatypes.complex64_

    check(False, lambda: f"Unknown type {typ}!")


def get_numberlike_type(x):
    if isinstance(x, NumberProxy):
        return x.python_type

    if isinstance(x, Number):
        return type(x)

    check(True, lambda: f"Unexpected type {type(x)}!")


def get_numberlike_value(x):
    if isinstance(x, NumberProxy):
        return x.value

    if isinstance(x, Number):
        return x

    check(True, lambda: f"Unexpected type {type(x)}!")


# TODO: maybe support numbers, too?
def check_same_dtype(*args):
    """Accepts multiple dtypes, TensorProxies, and numbers.

    Checks that all given dtypes and dtypes of TensorProxies are equivalent,
    and that all numbers have the corresponding Python type.

    Raises a RuntimeError otherwise.
    """

    if len(args) == 0:
        return None, None

    number_type = None
    tensor_dtype = None
    for a in args:
        if isinstance(a, Number):
            typ = get_numberlike_type(a)
            check(
                number_type is None or number_type is typ,
                lambda: f"Expected type {number_type} but found {typ}!",
            )
            number_type = typ
        else:
            dtype = _extract_dtype(a)
            check(
                tensor_dtype is None or tensor_dtype is dtype,
                lambda: f"Expected dtype {tensor_dtype} but found {dtype}!",
            )
            tensor_dtype = dtype

    if number_type is not None and tensor_dtype is not None:
        expected = dtype_to_type(tensor_dtype)
        check(
            number_type is expected,
            lambda: (
                f"Expected the type {expected}, corresponding to the dtype {tensor_dtype}, " f"but found {number_type}!"
            ),
        )

    return number_type, tensor_dtype


b8_, b8 = datatypes.bool8_, datatypes.bool8
u8_, u8 = datatypes.uint8_, datatypes.uint8
i8_, i8 = datatypes.int8_, datatypes.int8
i16_, i16 = datatypes.int16_, datatypes.int16
i32_, i32 = datatypes.int32_, datatypes.int32
i64_, i64 = datatypes.int64_, datatypes.int64

_exact_dtype_to_number_map = {
    bool: 0,
    b8_: 1,
    b8: 2,
    int: 3,
    u8_: 4,
    u8: 5,
    i8_: 6,
    i16_: 7,
    i32_: 8,
    i64_: 9,
    i8: 10,
    i16: 11,
    i32: 12,
    i64: 13,
}

# fmt: off
# Exact type lattice
#    b8_ -> b8 \
# b /-> i -> i8_ -> i16_ -> i32_ -> i64_ -> i8 -> i16 -> i32 -> i64
#                `-> u8_ -> u8 ----------------------------^
# TODO REVIEW: it's a little odd that u8_ + i64_ -> i16
_elementwise_exact_promotion_table = [
    #    b     b8_     b8      i    u8_    u8    i8_  i16_  i32_  i64_  i8   i16  i32  i64
    [ bool,    b8_,   b8,    int,  u8_,   u8,   i8_, i16_, i32_, i64_,  i8, i16, i32, i64], # b
    [  b8_,    b8_,   b8,    i8_,  u8_,   u8,   i8_, i16_, i32_, i64_,  i8, i16, i32, i64], # b8_
    [   b8,     b8,   b8,    i8_,  u8_,   u8,   i8_, i16_, i32_, i64_,  i8, i16, i32, i64], # b8
    [  int,    i8_,  i8_,    int,  u8_,   u8,   i8_, i16_, i32_, i64_,  i8, i16, i32, i64], # i
    [  u8_,    u8_,  u8_,    u8_,  u8_,   u8,   i16,  i16,  i16,  i16, i16, i16, i32, i64], # u8_
    [   u8,     u8,   u8,     u8,   u8,   u8,   i16,  i16,  i16,  i16, i16, i16, i32, i64], # u8
    [  i8_,    i8_,  i8_,    i8_,  i16,   i16,  i8_, i16_, i32_, i64_,  i8, i16, i32, i64], # i8_
    [ i16_,   i16_, i16_,   i16_,  i16,   i16, i16_, i16_, i32_, i64_,  i8, i16, i32, i64], # i16_
    [ i32_,   i32_, i32_,   i32_,  i16,   i16, i32_, i32_, i32_, i64_,  i8, i16, i32, i64], # i32_
    [ i64_,   i64_, i64_,   i64_,  i16,   i16, i64_, i64_, i64_, i64_,  i8, i16, i32, i64], # i64_
    [   i8,     i8,   i8,     i8,  i16,   i16,   i8,   i8,   i8,   i8,  i8, i16, i32, i64], # i8
    [  i16,    i16,  i16,    i16,  i16,   i16,  i16,  i16,  i16,  i16, i16, i16, i32, i64], # i16
    [  i32,    i32,  i32,    i32,  i32,   i32,  i32,  i32,  i32,  i32, i32, i32, i32, i64], # i32
    [  i64,    i64,  i64,    i64,  i64,   i64,  i64,  i64,  i64,  i64, i64, i64, i64, i64], # i64
]



bf_,     bf =  datatypes.bfloat16_,   datatypes.bfloat16
f16_,   f16 =  datatypes.float16_,    datatypes.float16
f32_,   f32 =  datatypes.float32_,    datatypes.float32
f64_,   f64 =  datatypes.float64_,    datatypes.float64
c32_,   c32 =  datatypes.complex32_,  datatypes.complex32
c64_,   c64 =  datatypes.complex64_,  datatypes.complex64
c128_, c128 =  datatypes.complex128_, datatypes.complex128

_inexact_dtype_to_number_map = {
    float   : 0,
    bf_     : 1,
    f16_    : 2,
    f32_    : 3,
    f64_    : 4,
    bf      : 5,
    f16     : 6,
    f32     : 7,
    f64     : 8,
    complex : 9,
    c32_    : 10,
    c64_    : 11,
    c128_   : 12,
    c32     : 13,
    c64     : 14,
    c128    : 15,
}

# Inexact type lattice
#    c* -> c32* -> c64* -> c128* -> c32 ----> c64 ----> c128
#   /    /        /       /       /          /        /
#  /    /        /       /   ,-> float16 -> fp32 -> fp64
# f -> fp16* -> fp32* -> fp64* -> bfloat16 --^
#  `-> bfloat16* -^
_elementwise_inexact_promotion_table = [
    #       f    bf_   f16_   f32_    f64_   bf   f16   f32   f64  complex   c32_   c64_  c128_   c32   c64  c128
    [   float,   bf_,  f16_,  f32_,  f64_,   bf,  f16,  f32,  f64, complex,  c32_,  c64_, c128_,  c32,  c64, c128], # f
    [     bf_,   bf_,  f32_,  f32_,  f64_,   bf,  f16,  f32,  f64,    c64_,  c32_,  c64_, c128_,  c32,  c64, c128], # bf_
    [    f16_,  f32_,  f16_,  f32_,  f64_,   bf,  f16,  f32,  f64,    c32_,  c32_,  c64_, c128_,  c32,  c64, c128], # f16_
    [    f32_,  f32_,  f32_,  f32_,  f64_,   bf,  f16,  f32,  f64,    c64_,  c32_,  c64_, c128_,  c32,  c64, c128], # f32_
    [    f64_,  f64_,  f64_,  f64_,  f64_,   bf,  f16,  f32,  f64,   c128_,  c32_,  c64_, c128_,  c32,  c64, c128], # f64_
    [      bf,    bf,    bf,    bf,    bf,   bf,  f32,  f32,  f64,     c64,   c64,   c64,   c64,  c64,  c64, c128], # bf
    [     f16,   f16,   f16,   f16,   f16,  f32,  f16,  f32,  f64,     c32,   c32,   c32,   c32,  c32,  c64, c128], # f16
    [     f32,   f32,   f32,   f32,   f32,  f32,  f32,  f32,  f64,     c64,   c64,   c64,   c64,  c64,  c64, c128], # f32
    [     f64,   f64,   f64,   f64,   f64,  f64,  f64,  f64,  f64,    c128,  c128,  c128,  c128, c128, c128, c128], # f64
    [ complex,  c64_,  c32_,  c64_, c128_,  c64,  c32,  c64, c128, complex,  c32_,  c64_, c128_,  c32,  c64, c128], # complex
    [    c32_,  c64_,  c32_,  c64_, c128_,  c64,  c32,  c64, c128,    c32_,  c32_,  c64_, c128_,  c32,  c64, c128], # c32_
    [    c64_,  c64_,  c64_,  c64_, c128_,  c64,  c32,  c64, c128,    c64_,  c64_,  c64_, c128_,  c32,  c64, c128], # c64_
    [   c128_, c128_, c128_, c128_, c128_,  c64,  c32,  c64, c128,   c128_, c128_, c128_, c128_,  c32,  c64, c128], # c128_
    [     c32,   c32,   c32,   c32,   c32,  c64,  c32,  c64, c128,     c32,   c32,   c32,   c32,  c32,  c64, c128], # c32
    [     c64,   c64,   c64,   c64,   c64,  c64,  c64,  c64, c128,     c64,   c64,   c64,   c64,  c64,  c64, c128], # c64
    [    c128,  c128,  c128,  c128,  c128, c128, c128, c128, c128,    c128,  c128,  c128,  c128, c128, c128, c128], # c128
]
# fmt: on


def _elementwise_type_promotion(a, b):
    # Inexact x exact and exact x inexact cases
    # Inexact dtypes take preference over exact dtypes
    if is_inexact_dtype(a) and is_exact_dtype(b):
        return a
    if is_exact_dtype(a) and is_inexact_dtype(b):
        return b

    # Exact x Exact case
    # b -> i -> i8* -> i16* -> i32* -> i64* -> i8 -> i16 -> i32 -> i64
    #       `-> u8* -> u8 ----------------------------^
    if is_exact_dtype(a):
        a_idx, b_idx = _exact_dtype_to_number_map[a], _exact_dtype_to_number_map[b]
        return _elementwise_exact_promotion_table[a_idx][b_idx]

    # Inexact x Inexact case
    # c* -> c32* -> c64* -> c128* -> c32 ----> c64 ----> c128
    #       /        /       /       /          /        /
    #      /        /       /   ,-> float16 -> fp32 -> fp64
    # fp16* ---> fp32* -> fp64* -> bfloat16 --^
    # bfloat16* -^
    a_idx, b_idx = _inexact_dtype_to_number_map[a], _inexact_dtype_to_number_map[b]
    return _elementwise_inexact_promotion_table[a_idx][b_idx]


# Maps datatypes to their computation types for elementwise operations
_computation_dtype_map = {
    datatypes.float16_: datatypes.float32_,
    datatypes.float16: datatypes.float32,
    datatypes.bfloat16_: datatypes.float32_,
    datatypes.bfloat16: datatypes.float32,
    datatypes.complex32_: datatypes.complex64_,
    datatypes.complex32: datatypes.complex64,
}


def _computation_dtype(dtype):
    return _computation_dtype_map.get(dtype, dtype)


def _has_weak_dtype(x):
    return x.weak_dtype or is_number_tensor(x)


class ELEMENTWISE_TYPE_PROMOTION_KIND(Enum):
    DEFAULT = (0,)
    PRESERVE = (1,)
    INT_TO_FLOAT = (2,)
    ALWAYS_BOOL = (3,)
    COMPLEX_TO_FLOAT = (4,)
    BOOL_TO_LONG = (5,)


# TODO: generalize to varargs, allow numbers, dtypes, and tensors
def elementwise_type_promotion(*args, type_promotion_kind: ELEMENTWISE_TYPE_PROMOTION_KIND):
    """Computes the computation and result dtypes for elementwise type promotion on the given arguments and with the
    given elementwise type promotion kind.

    Type promotion in Thunder conceptually corresponds with JAX's type promotion.
    See https://jax.readthedocs.io/en/latest/type_promotion.html.

    Reviewing the inputs determines a "promotion dtype", but this function returns a
    "computation dtype" and a "result dtype." The "type_promotion"kind" argument
    determines how the promotion dtype is mapped to a computation and result dtype.

    PRESERVE preserves the promotion dtype as the computation and result dtype.
    It's appropriate for kernels that perform no mathematical operations on their tensors.

    DEFAULT type promotion selects a computation dtype by mapping low precision promotion dtypes to their
    higher precision counterparts:

      float16   -> float32
      bfloat16  -> float32
      complex32 -> complex64

    The result dtype is the same as the promotion dtype.

    INT_TO_FLOAT is like DEFAULT, except integer promotion dtypes map to float for their
    computation and result dtypes.

    COMPLEX_TO_FLOAT is like DEFAULT, except complex promotion dtypes have their corresponding
    float dtypes as a return dtype:

        complex32  -> float16
        complex64  -> float32
        complex128 -> float64

    BOOL_TO_LONG is like DEFAULT, except boolean promotion types use int64 for their computation
    and result dtypes.

    ALWAYS_BOOL is like PRESERVE, except the result dtype is always bool.

    Example operators for each type promotion option:

      DEFAULT                 : add
      PRESERVE                : where, nextafter, cat
      INT_TO_FLOAT            : sin
      COMPLEX_TO_FLOAT        : abs
      BOOL_TO_LONG            : pow
      ALWAYS_BOOL             : eq
    """

    # Type checks inputs
    assert all(isinstance(a, (TensorProxy, Number)) for a in args)
    assert len(args) > 0

    has_tensor_input = False
    extracted = []
    for a in args:
        if isinstance(a, Number):
            extracted.append(get_numberlike_type(a))
        else:
            # x is a TensorProxy
            has_tensor_input = True
            extracted.append(a.dtype)

    promotion_dtype = extracted[0]
    for dtype in extracted[1:]:
        promotion_dtype = _elementwise_type_promotion(promotion_dtype, dtype)
        if has_tensor_input:
            promotion_dtype = type_to_dtype(promotion_dtype)

    if type_promotion_kind is ELEMENTWISE_TYPE_PROMOTION_KIND.PRESERVE:
        return promotion_dtype, promotion_dtype

    if type_promotion_kind is ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL:
        if has_tensor_input:
            return promotion_dtype, datatypes.bool8
        return promotion_dtype, bool

    if type_promotion_kind is ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT and is_integer_dtype(promotion_dtype):
        if has_tensor_input:
            return datatypes.float32_, datatypes.float32_
        return float, float

    if type_promotion_kind is ELEMENTWISE_TYPE_PROMOTION_KIND.COMPLEX_TO_FLOAT and is_complex_dtype(promotion_dtype):
        return (
            _computation_dtype(promotion_dtype),
            corresponding_real_dtype(promotion_dtype),
        )

    if type_promotion_kind is ELEMENTWISE_TYPE_PROMOTION_KIND.BOOL_TO_LONG and is_boolean_dtype(promotion_dtype):
        if has_tensor_input:
            return datatypes.int64_, datatypes.int64_
        return int, int

    # Falls through to DEFAULT
    return _computation_dtype(promotion_dtype), promotion_dtype


#
# Shape-related functions
#


def is_number_tensor(t):
    """True if the input is a "number tensor" -- a single element tensor with an empty shape.

    False otherwise.
    """
    if len(t.shape) == 0:
        return True

    return False


# TODO: maybe generalize to *args like check_same_dtype
# TODO: change to check_same_shape or add check_same_shape variant and make check_same_dtype use the same pattern
def same_shape(a, b):
    if len(a) != len(b):
        return False

    for x, y in zip(a, b):
        if x != y:
            return False

    return True


# "Wraps" a dim (up to one time) for the given rank, allowing dims to be
# specified using negative indices. For scalar tensors with rank 0, then idx
# must be in the range [-1, 0]. Otherwise, idx should be in the range [-rank, rank-1].
def canonicalize_dim(rank: int, idx: int, wrap_scalar: bool = True) -> int:
    check(rank >= 0, lambda: f"Rank cannot be negative but got {rank}!")

    if rank == 0:
        check(
            wrap_scalar,
            lambda: f"Dimension specified as {idx} but tensor has no dimensions!",
            exception_type=IndexError,
        )
        rank = 1

    if idx >= 0 and idx < rank:
        return idx

    if idx < 0:
        _idx = idx + rank
    else:
        _idx = idx

    check(
        _idx >= 0 and _idx < rank,
        lambda: f"Dimension out of range (expected to be in range of [{-rank}, {rank - 1}], but got {idx})",
        exception_type=IndexError,
    )

    return _idx


def check_valid_length(length: int):
    """Validates that an object represents a valid dimension length."""

    check(length >= 0, lambda: f"Found invalid length {length}!")


def check_valid_shape(shape):
    """Validates that a sequence represents a valid shape."""

    for l in shape:
        check_valid_length(l)


def validate_idx(rank: int, idx: int):
    """Validates that idx is a valid index for the given shape.

    Assumes the index is already canonicalized.
    """

    check(
        idx >= 0 and (idx < rank or idx == 0),
        lambda: f"Found invalid index {idx} for rank {rank}!",
    )


def check_no_duplicates(dims: Sequence):
    check(len(dims) == len(set(dims)), lambda: f"Duplicate value in list of dimensions {dims}!")


# TODO: think about preserving the original function's signature
class langctx(object):
    """
    A decorator that calls the decorated function in the given language context,
    resetting to the caller's language context when the function is done.
    """

    def __init__(self, ctx):
        self.ctx = ctx

    def __call__(self, fn_):
        @wraps(fn_)
        def fn(*args, **kwargs):
            tok = trace.set_language_context(self.ctx)
            result = fn_(*args, **kwargs)
            trace.set_language_context(tok)
            return result

        return fn
