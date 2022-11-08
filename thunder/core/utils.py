from typing import Callable, List, Type, Tuple, Union
from numbers import Number

from .proxies import TensorProxy, NumberProxy, NumberLike

# TODO: remove unconditional torch import
import torch

# This file defines utilities that can be used when defining primitive operations.

# This file depends on proxies.py.

__all__ = [
    # Error checking helpers
    "check",
    # Datatype and Python type-related functions
    "check_same_dtype",
    "get_numberlike_type",
    "get_numberlike_value",
    "type_to_dtype",
    # Shape-related functions
    "same_shape",
    "check_valid_length",
    "check_valid_shape",
]

# Common types
# TODO: extend types if libraries like torch are available
ShapeType = Union[List[int], Tuple[int, ...]]


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
_integer_dtypes = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
_low_precision_dtypes = (torch.float16, torch.bfloat16, torch.complex32)
_float_dtypes = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
_complex_dtypes = (torch.complex32, torch.complex64, torch.complex128)


def dtype_to_type(dtype: torch.dtype) -> type:
    """
    Computes the corresponding Python type (AKA "type kind") for the
    given dtype.
    """
    assert isinstance(dtype, torch.dtype)

    if dtype is torch.bool:
        return bool
    if dtype in _integer_dtypes:
        return int
    if dtype in _float_dtypes:
        return float
    if dtype in _complex_dtypes:
        return complex


_real_to_complex_dtype_map = {
    torch.float16: torch.complex32,
    torch.bfloat16: torch.complex64,
    torch.float32: torch.complex64,
    torch.float64: torch.complex128,
}


def corresponding_complex_dtype(dtype: torch.dtype) -> torch.dtype:
    return _real_to_complex_dtype_map[dtype]


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
    Accepts multiple torch.dtypes, objects with a 'dtype' property.

    Checks that all given torch.dtypes or obj.dtype properties are equivalent.
    Raises a RuntimeError otherwise.
    """

    if len(args) == 0:
        return

    def _extract_dtype(x):
        if isinstance(x, torch.dtype):
            return x

        if hasattr(x, "dtype"):
            return x.dtype

        raise AssertionError(f"Trying to extract dtype from unknown type {x.dtype}!")

    dtypes = tuple(map(_extract_dtype, args))

    expected_dtype = dtypes[0]
    for dtype in dtypes[1:]:
        check(
            dtype == expected_dtype,
            lambda: f"Found distinct dtype {dtype}, expected {expected_dtype}!",
        )


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


def check_valid_length(length: int):
    """
    Validates that an object represents a valid
    dimension length.
    """

    assert length >= 0


def check_valid_shape(shape: ShapeType):
    """
    Validates that a sequence represents a valid shape.
    """

    for l in shape:
        check_valid_length(l)
