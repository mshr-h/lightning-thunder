from typing import Callable, List, Type, Tuple, Union, Sequence
from numbers import Number
from enum import Enum

from .proxies import TensorProxy, NumberProxy

# TODO: remove unconditional torch import
import torch

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
    "is_number_type",
    "corresponding_real_dtype",
    "corresponding_complex_dtype",
    "dtype_to_type",
    "type_to_dtype",
    "check_same_dtype",
    "get_numberlike_type",
    "get_numberlike_value",
    "ELEMENTWISE_TYPE_PROMOTION_KIND",
    "elementwise_type_promotion",
    # Shape-related functions
    "same_shape",
    "canonicalize_dim",
    "check_valid_length",
    "check_valid_shape",
    "check_valid_index",
    "check_no_duplicates",
]

# Common types
# TODO: extend types if libraries like torch are available
ShapeType = Union[List[int], Tuple[int, ...]]

# TODO: Creates type.py and let thunder.Tensor stand for any tensor object


#
# Error checking helpers
#

# TODO: maybe put check in a dependency-free base_utils that's also imported here (so it can be used by proxies.py)
def check(
    cond: bool, s: Callable[[], str], exception_type: Type[Exception] = RuntimeError
) -> None:
    """
    Helper function for raising an error_type (default: RuntimeError) if a boolean condition fails.

    s is a callable producing a string to avoid string construction if the error check is passed.
    """
    if not cond:
        raise exception_type(s())


#
# Datatype-related functions
#

# TODO: make this non-Torch-specific
_integer_dtypes = (
    torch.bool,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
)
_low_precision_dtypes = (torch.float16, torch.bfloat16, torch.complex32)
_float_dtypes = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
_complex_dtypes = (torch.complex32, torch.complex64, torch.complex128)


def is_boolean_dtype(dtype: torch.dtype) -> bool:
    return dtype is torch.bool or dtype is bool


def is_integer_dtype(dtype) -> bool:
    return dtype in _integer_dtypes or dtype in (bool, int)


def is_low_precision_dtype(dtype: torch.dtype) -> bool:
    return dtype in _low_precision_dtypes


def is_float_dtype(dtype: torch.dtype) -> bool:
    return dtype in _float_dtypes or dtype is float


def is_complex_dtype(dtype: torch.dtype) -> bool:
    return dtype in _complex_dtypes or dtype is complex


def is_number_type(typ):
    return typ in (bool, int, float, complex)


_complex_to_real_dtype_map = {
    torch.complex128: torch.float64,
    torch.complex64: torch.float32,
    torch.complex32: torch.float16,
}

_real_to_complex_dtype_map = {
    torch.float16: torch.complex32,
    torch.bfloat16: torch.complex64,
    torch.float32: torch.complex64,
    torch.float64: torch.complex128,
}


def corresponding_real_dtype(dtype: torch.dtype) -> torch.dtype:
    return _complex_to_real_dtype_map[dtype]


def corresponding_complex_dtype(dtype: torch.dtype) -> torch.dtype:
    return _real_to_complex_dtype_map[dtype]


def dtype_to_type(dtype: torch.dtype) -> type:
    """
    Computes the corresponding Python type (AKA "type kind") for the
    given dtype.
    """
    if is_boolean_dtype(dtype):
        return bool
    if is_integer_dtype(dtype):
        return int
    if is_float_dtype(dtype):
        return float
    if is_complex_dtype(dtype):
        return complex

    check(False, lambda: f"Unknown dtype {dtype}!")


# TODO: probably want to remove this by treating the python number types as dtypes
def type_to_dtype(typ: type) -> torch.dtype:
    """
    Computes the corresponding dtype for a Number type.
    """

    assert isinstance(typ, type)

    if typ is bool:
        return torch.bool
    if typ is int:
        return torch.long
    if typ is float:
        return torch.get_default_dtype()
    if typ is complex:
        return corresponding_complex_dtype(torch.get_default_dtype())

    raise ValueError("Invalid type {typ}!")


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
    """
    Accepts multiple torch.dtypes, objects with a 'dtype' property, and numbers.

    Checks that all given torch.dtypes or obj.dtype properties are equivalent,
    and that all numbers have the corresponding Python type.

    Raises a RuntimeError otherwise.
    """

    if len(args) == 0:
        return None, None

    def _extract_dtype(x):
        if isinstance(x, torch.dtype):
            return x

        if hasattr(x, "dtype"):
            return x.dtype

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
            lambda: f"Expected the type {expected}, corresponding to the dtype {tensor_dtype}, but found {number_type}!",
        )

    return number_type, tensor_dtype


# TODO: construct Thunder types that include i_, f_, and c_
# TODO: consider more efficient datastructures
b1, u1, i1, i2, i4, i8 = _integer_dtypes
f2, bf, f4, f8 = _float_dtypes
c4, c8, c16 = _complex_dtypes
b_, i_, f_, c_ = bool, int, float, complex

_dtype_to_number_map = {
    # Note: b_ and b1 are currently synonymous
    # TODO: canonicalize to just bool to properly support bool x bool operations
    b_: 0,
    b1: 0,
    u1: 1,
    i1: 2,
    i2: 3,
    i4: 4,
    i8: 5,
    bf: 6,
    f2: 7,
    f4: 8,
    f8: 9,
    c4: 10,
    c8: 11,
    c16: 12,
    i_: 13,
    f_: 14,
    c_: 15,
}

# TODO: manually reformat this table
# fmt: off
_elementwise_promotion_table = [
    # b1   u1   i1	 i2	  i4   i8   bf	 f2	  f4   f8   c4   c8  c16   i_   f_   c_
    [b1, u1, i1, i2, i4, i8, bf, f2, f4, f8, c4, c8, c16, i_, f_, c_],  # b1
    [u1, u1, i2, i2, i4, i8, bf, f2, f4, f8, c4, c8, c16, u1, f_, c_],  # u1
    [i1, i2, i1, i2, i4, i8, bf, f2, f4, f8, c4, c8, c16, i1, f_, c_],  # i1
    [i2, i2, i2, i2, i4, i8, bf, f2, f4, f8, c4, c8, c16, i2, f_, c_],  # i2
    [i4, i4, i4, i4, i4, i8, bf, f2, f4, f8, c4, c8, c16, i4, f_, c_],  # i4
    [i8, i8, i8, i8, i8, i8, bf, f2, f4, f8, c4, c8, c16, i8, f_, c_],  # i8
    [bf, bf, bf, bf, bf, bf, bf, f4, f4, f8, c8, c8, c16, bf, bf, c8],  # bf
    [f2, f2, f2, f2, f2, f2, f4, f2, f4, f8, c4, c8, c16, i2, f2, c4],  # f2
    [f4, f4, f4, f4, f4, f4, f4, f4, f4, f8, c8, c8, c16, f4, f4, c8],  # f4
    [f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, c16, c16, c16, f8, f8, c16],  # f8
    [c4, c4, c4, c4, c4, c4, c8, c4, c8, c16, c4, c8, c16, c4, c4, c4],  # c4
    [c8, c8, c8, c8, c8, c8, c8, c8, c8, c16, c8, c8, c16, c8, c8, c8],  # c8
    [
        c16,
        c16,
        c16,
        c16,
        c16,
        c16,
        c16,
        c16,
        c16,
        c16,
        c16,
        c16,
        c16,
        c16,
        c16,
        c16,
    ],  # c16
    [i_, u1, i1, i2, i4, i8, bf, f2, f4, f8, c4, c8, c16, i_, f_, c_],  # i_
    [f_, f_, f_, f_, f_, f_, bf, f2, f4, f8, c4, c8, c16, f_, f_, c_],  # f_
    [c_, c_, c_, c_, c_, c_, c8, c4, c8, c16, c4, c8, c16, c_, c_, c_],  # c_
    # b1   u1   i1	 i2	  i4   i8   bf	 f2	  f4   f8   c4   c8  c16   i_   f_   c_
]
# fmt: on

# TODO: working here
# map types/dtypes to numbers that represent table position
# lookup in table
def _elementwise_type_promotion(a, b):
    a, b = _dtype_to_number_map[a], _dtype_to_number_map[b]
    return _elementwise_promotion_table[a][b]


# Maps datatypes to their computation types for elementwise operations
_computation_dtype_map = {
    torch.bfloat16: torch.float32,
    torch.float16: torch.float32,
    torch.complex32: torch.complex64,
}


def _computation_dtype(dtype):
    return _computation_dtype_map.get(dtype, dtype)


class ELEMENTWISE_TYPE_PROMOTION_KIND(Enum):
    DEFAULT = (0,)
    PRESERVE = (1,)
    INT_TO_FLOAT = (2,)
    ALWAYS_BOOL = (3,)
    COMPLEX_TO_FLOAT = (4,)
    BOOL_TO_LONG = (5,)


# TODO: generalize to varargs, allow numbers, dtypes, and tensors
def elementwise_type_promotion(
    a, b, type_promotion_kind: ELEMENTWISE_TYPE_PROMOTION_KIND
):
    """
    Computes the computation and result dtypes for elementwise type promotion
    on the given arguments and with the given elementwise type promotion kind.

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

    assert isinstance(a, (TensorProxy, Number))
    assert isinstance(b, (TensorProxy, Number))

    # TODO: update to handle wealky typed tensors
    def _extract_type_or_dtype(x):
        if isinstance(x, Number):
            return get_numberlike_type(x)

        # x is a TensorProxy
        return x.dtype

    a_dtype, b_dtype = _extract_type_or_dtype(a), _extract_type_or_dtype(b)
    promotion_dtype = _elementwise_type_promotion(a_dtype, b_dtype)

    if type_promotion_kind is ELEMENTWISE_TYPE_PROMOTION_KIND.PRESERVE:
        return promotion_dtype, promotion_dtype

    if type_promotion_kind is ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL:
        return promotion_dtype, torch.bool

    if (
        type_promotion_kind is ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
        and is_integer_dtype(promotion_dtype)
    ):
        return float, float

    if (
        type_promotion_kind is ELEMENTWISE_TYPE_PROMOTION_KIND.COMPLEX_TO_FLOAT
        and is_complex_dtype(promotion_dtype)
    ):
        return _computation_dtype(promotion_dtype), corresponding_real_dtype(
            promotion_dtype
        )

    if (
        type_promotion_kind is ELEMENTWISE_TYPE_PROMOTION_KIND.BOOL_TO_LONG
        and is_boolean_dtype(promotion_dtype)
    ):
        return torch.long, torch.long

    # Falls through to DEFAULT
    return _computation_dtype(promotion_dtype), promotion_dtype


#
# Shape-related functions
#

# TODO: maybe generalize to *args like check_same_dtype
# TODO: change to check_same_shape or add check_same_shape variant and make check_same_dtype use the same pattern
def same_shape(a: ShapeType, b: ShapeType) -> bool:
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
    """
    Validates that an object represents a valid
    dimension length.
    """

    check(length >= 0, lambda: f"Found invalid length {length}!")


def check_valid_shape(shape: ShapeType):
    """
    Validates that a sequence represents a valid shape.
    """

    for l in shape:
        check_valid_length(l)


def validate_idx(rank: int, idx: int):
    """
    Validates that idx is a valid index for the given shape.
    Assumes the index is already canonicalized.
    """

    check(
        idx >= 0 and (idx < rank or idx == 0),
        lambda: f"Found invalid index {idx} for rank {rank}!",
    )


def check_no_duplicates(dims: Sequence):
    check(len(dims) == len(set(dims)), lambda: f"Duplicate value in {dims}!")
