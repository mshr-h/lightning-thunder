from typing import Callable, Sequence, Type
from functools import reduce

from . import prims
from . import utils

# This files defines Thunder's core operators.
# These operators are distinct from Thunder's primitives, which are the building blocks to build languages
# from. These operators are build using primitives, and are intended to make it easier to write
# other language definitions using Thunder.

# This file depends on prims.py.

__all__ = [
    # Shape operations
    "expand",
    # Elemenwise unary operations
    "abs",
    # Elementwise binary operations
    "add",
]

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

    return prims.broadcast_in_dim(
        a, shape_, tuple(range(offset, len(a.shape) + offset))
    )


def _compute_broadcast_shape(*_shapes):
    """
    Computes the common shape with the fewest dimensions that all input shapes can be broadcast to.
    """
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
                f"Attempting to broadcast a dimension of length {shape[idx]}!",
            )

    return tuple(common_shape)


# TODO: add scalar support
# TODO: review hasattr pattern
def _maybe_broadcast(*args):
    """
    Returns tensors with the same shape, possibly broadcasting inputs to the result shape.
    """

    # Computes common shape
    common_shape = _compute_broadcast_shape(
        *map(lambda t: t.shape if hasattr(t, "shape") else None, args)
    )

    def __maybe_broadcast(x, shape):
        if hasattr(x, "shape"):
            if not utils.same_shape(x.shape, common_shape):
                return expand(x, common_shape)

        return x

    return tuple(__maybe_broadcast(x, common_shape) for x in args)


#
# Elementwise unary operations
#
def abs(a):
    return prims.abs(a)


#
# Elementwise binary operations
#

# TODO: add type promotion
# TODO: add scalar support
def add(a, b):
    a, b = _maybe_broadcast(a, b)

    return prims.add(a, b)
