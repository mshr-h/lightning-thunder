from enum import Enum
from numbers import Number
from typing import Callable, Sequence, Type

import thunder.core.dtypes as dtypes

from .proxies import NumberProxy, TensorProxy

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

# TODO: review these sequences after refactoring Thunder dtypes
#   ... probably want to add metadata to singleton dtype objects
_integer_dtypes = (
    dtypes.uint8_,
    dtypes.uint8,
    dtypes.int8_,
    dtypes.int8,
    dtypes.int16_,
    dtypes.int16,
    dtypes.int32_,
    dtypes.int32,
    dtypes.int64_,
    dtypes.int64,
)

_low_precision_dtypes = (
    dtypes.float16_,
    dtypes.float16,
    dtypes.bfloat16_,
    dtypes.bfloat16,
    dtypes.complex32_,
    dtypes.complex32,
)

_float_dtypes = (
    dtypes.float16_,
    dtypes.float16,
    dtypes.bfloat16_,
    dtypes.bfloat16,
    dtypes.float32_,
    dtypes.float32,
    dtypes.float64_,
    dtypes.float64,
)

_complex_dtypes = (
    dtypes.complex32_,
    dtypes.complex32,
    dtypes.complex64_,
    dtypes.complex64,
    dtypes.complex128_,
    dtypes.complex128,
)


def is_boolean_dtype(dtype) -> bool:
    return dtype is bool


def is_unsigned_dtype(dtype):
    return dtype in (bool, dtypes.uint8_, dtypes.uint8)


def is_signed_integer_dtype(dtype):
    return is_integer_dtype(dtype) and not is_unsigned_dtype(dtype)


def is_integer_dtype(dtype) -> bool:
    return dtype in _integer_dtypes or dtype in (bool, int)


is_exact_dtype = is_integer_dtype


def is_low_precision_dtype(dtype) -> bool:
    return dtype in _low_precision_dtypes


def is_float_dtype(dtype) -> bool:
    return dtype in _float_dtypes or dtype is float


def is_complex_dtype(dtype) -> bool:
    return dtype in _complex_dtypes or dtype is complex


def is_inexact_dtype(dtype):
    return is_float_dtype(dtype) or is_complex_dtype(dtype)


def is_number_type(typ):
    return typ in (bool, int, float, complex)


def is_weak_dtype(dtype):
    return dtype in (
        bool,
        int,
        float,
        complex,
        dtypes.uint8_,
        dtypes.int8_,
        dtypes.int16_,
        dtypes.int32_,
        dtypes.int64_,
        dtypes.bfloat16_,
        dtypes.float16_,
        dtypes.float32_,
        dtypes.float64_,
        dtypes.complex32_,
        dtypes.complex64_,
        dtypes.complex128_,
    )


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


_complex_to_real_dtype_map = {
    dtypes.complex128_: dtypes.float64_,
    dtypes.complex128: dtypes.float64,
    dtypes.complex64_: dtypes.float32_,
    dtypes.complex64: dtypes.float32,
    dtypes.complex32_: dtypes.float16_,
    dtypes.complex32: dtypes.float16,
    complex: float,
}

_real_to_complex_dtype_map = {
    dtypes.bfloat16_: dtypes.complex64_,
    dtypes.bfloat16: dtypes.complex64,
    dtypes.float16_: dtypes.complex32_,
    dtypes.float16: dtypes.complex32,
    dtypes.float32_: dtypes.complex64_,
    dtypes.float32: dtypes.complex64,
    dtypes.float64_: dtypes.complex128_,
    dtypes.float64: dtypes.complex128,
    float: complex,
}


def corresponding_real_dtype(dtype):
    return _complex_to_real_dtype_map[dtype]


def corresponding_complex_dtype(dtype):
    return _real_to_complex_dtype_map[dtype]


_dtype_to_weak_dtype_map = {
    dtypes.uint8: dtypes.uint8_,
    dtypes.int8: dtypes.int8_,
    dtypes.int16: dtypes.int16_,
    dtypes.int32: dtypes.int32_,
    dtypes.int64: dtypes.int64_,
    dtypes.bfloat16: dtypes.bfloat16_,
    dtypes.float16: dtypes.float16_,
    dtypes.float32: dtypes.float32_,
    dtypes.float64: dtypes.float64_,
    dtypes.complex32: dtypes.complex32_,
    dtypes.complex64: dtypes.complex64_,
    dtypes.complex128: dtypes.complex128_,
}


def corresponding_weak_dtype(dtype):
    if is_weak_dtype(dtype):
        return dtype
    return _dtype_to_weak_dtype_map[dtype]


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

    def _extract_dtype(x):
        if isinstance(x, dtypes.datatype):
            return x

        if isinstance(x, TensorProxy):
            return x.thunder_dtype()

        raise AssertionError(f"Trying to extract dtype from unknown type {x}!")

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


u8_, u8 = dtypes.uint8_, dtypes.uint8
i8_, i8 = dtypes.int8_, dtypes.int8
i16_, i16 = dtypes.int16_, dtypes.int16
i32_, i32 = dtypes.int32_, dtypes.int32
i64_, i64 = dtypes.int64_, dtypes.int64

_exact_dtype_to_number_map = {
    bool: 0,
    int: 1,
    u8_: 2,
    u8: 3,
    i8_: 4,
    i16_: 5,
    i32_: 6,
    i64_: 7,
    i8: 8,
    i16: 9,
    i32: 10,
    i64: 11,
}

# fmt: off
# Exact type lattice
# b -> i -> i8_ -> i16_ -> i32_ -> i64_ -> i8 -> i16 -> i32 -> i64
#       `-> u8_ -> u8 ----------------------------^
# TODO REVIEW: it's a little odd that u8_ + i64_ -> i16
_elementwise_exact_promotion_table = [
    #    b      i   u8_     u8    i8_   i16_   i32_  i64_  i8  i16  i32  i64
    [ bool,   int,  u8_,    u8,   i8_,  i16_,  i32_, i64_,  i8, i16, i32, i64], # b
    [  int,   int,  u8_,    u8,   i8_,  i16_,  i32_, i64_,  i8, i16, i32, i64], # i
    [  u8_,   u8_,  u8_,    u8,   i16,   i16,   i16,  i16, i16, i16, i32, i64], # u8_
    [   u8,    u8,   u8,    u8,   i16,   i16,   i16,  i16, i16, i16, i32, i64], # u8
    [  i8_,   i8_,  i16,   i16,   i8_,  i16_,  i32_, i64_,  i8, i16, i32, i64], # i8_
    [ i16_,  i16_,  i16,   i16,  i16_,  i16_,  i32_, i64_,  i8, i16, i32, i64], # i16_
    [ i32_,  i32_,  i16,   i16,  i32_,  i32_,  i32_, i64_,  i8, i16, i32, i64], # i32_
    [ i64_,  i64_,  i16,   i16,  i64_,  i64_,  i64_, i64_,  i8, i16, i32, i64], # i64_
    [   i8,    i8,  i16,   i16,    i8,   i8,     i8,   i8,  i8, i16, i32, i64], # i8
    [  i16,   i16,  i16,   i16,   i16,  i16,    i16,  i16, i16, i16, i32, i64], # i16
    [  i32,   i32,  i32,   i32,   i32,  i32,    i32,  i32, i32, i32, i32, i64], # i32
    [  i64,   i64,  i64,   i64,   i64,  i64,    i64,  i64, i64, i64, i64, i64], # i64
]



bf_,     bf =  dtypes.bfloat16_,   dtypes.bfloat16
f16_,   f16 =  dtypes.float16_,    dtypes.float16
f32_,   f32 =  dtypes.float32_,    dtypes.float32
f64_,   f64 =  dtypes.float64_,    dtypes.float64
c32_,   c32 =  dtypes.complex32_,  dtypes.complex32
c64_,   c64 =  dtypes.complex64_,  dtypes.complex64
c128_, c128 =  dtypes.complex128_, dtypes.complex128

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
    dtypes.float16_: dtypes.float32_,
    dtypes.float16: dtypes.float32,
    dtypes.bfloat16_: dtypes.float32_,
    dtypes.bfloat16: dtypes.float32,
    dtypes.complex32_: dtypes.complex64_,
    dtypes.complex32: dtypes.complex64,
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

    def _extract_dtype(x):
        if isinstance(x, Number):
            return get_numberlike_type(x)

        # x is a TensorProxy
        has_tensor_input = True
        if is_number_tensor(x):
            return corresponding_weak_dtype(x.thunder_dtype())
        return x.thunder_dtype()

    extracted = tuple(_extract_dtype(a) for a in args)
    promotion_dtype = extracted[0]
    for dtype in extracted[1:]:
        promotion_dtype = _elementwise_type_promotion(promotion_dtype, dtype)

    if type_promotion_kind is ELEMENTWISE_TYPE_PROMOTION_KIND.PRESERVE:
        return promotion_dtype, promotion_dtype

    if type_promotion_kind is ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL:
        return promotion_dtype, bool

    if type_promotion_kind is ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT and is_integer_dtype(promotion_dtype):
        if not has_tensor_input:
            return dtypes.float32_, dtypes.float32_
        return float, float

    if type_promotion_kind is ELEMENTWISE_TYPE_PROMOTION_KIND.COMPLEX_TO_FLOAT and is_complex_dtype(promotion_dtype):
        return (
            _computation_dtype(promotion_dtype),
            corresponding_real_dtype(promotion_dtype),
        )

    if type_promotion_kind is ELEMENTWISE_TYPE_PROMOTION_KIND.BOOL_TO_LONG and is_boolean_dtype(promotion_dtype):
        if has_tensor_input:
            return dtypes.int64_, dtypes.int64_
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
    check(len(dims) == len(set(dims)), lambda: f"Duplicate value in {dims}!")
