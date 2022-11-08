from typing import Callable, List, Type, Tuple, Union

from .proxies import TensorProxy

# TODO: remove unconditional torch import
import torch

# This file defines utilities that can be used when defining primitive operations.

# This file depends on proxies.py.

__all__ = [
    # Functions related to error checking
    "check",
    "check_same_dtype",
    "same_shape",
]

# Common types
# TODO: extend types if libraries like torch are available
ShapeType = Union[List[int], Tuple[int, ...]]


#
# Functions related to error checking
#


def check(
    cond: bool, s: Callable[[], str], exception_type: Type[Exception] = RuntimeError
) -> None:
    """
    Helper function for raising an error_type (default: RuntimeError) if a boolean condition fails.

    s is a callable producing a string to avoid string construction if the error check is passed.
    """
    if not cond:
        raise exception_type(s())


# TODO: (re-)enable scalar support
def check_same_dtype(*args):
    """
    Accepts multiple torch.dtypes or objects with a 'dtype' property.

    Checks that all given torch.dtypes or obj.dtype properties are equivalent, and raises a
    RuntimeError otherwise.
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


# TODO: maybe generalize to *args like check_same_dtype
# TODO: change to check_same_shape or add check_same_shape variant and make check_same_dtype use the same pattern
def same_shape(a: ShapeType, b: ShapeType) -> bool:
    if len(a) != len(b):
        return False

    for x, y in zip(a, b):
        if x != y:
            return False

    return True


# TODO: revise with common check pattern
# TODO: improve error message
def check_valid_length(length: int):
    """
    Validates that an object represents a valid
    dimension length.
    """

    check(length >= 0, f"Invalid length {length}!")


def check_valid_shape(shape):
    """
    Validates that a sequence represents a valid shape.

    (That every element is a valid length.)
    """

    for l in shape:
        check_valid_length(l)
