import operator
from collections import deque

from .trace import Constraint, get_trace

# TODO: don't unconditionally import torch
import torch

# This file defines Thunder's most basic proxies, stand-ins for other Python objects that
#   record Python interactions for the tracing context.

# This file depends on trace.py.

__all__ = [
    "IntegerProxy",
    "TensorProxy",
    "proxy",
]

# TODO: consider subclassing or "fake subclassing" int
# See https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types
# TODO: this should pass isinstance(p, int)
class IntegerProxy(object):
    """
    A wrapper for an integer that records the dunder methods called on it and their
    results as constraints.

    TODO: implement all dunder methods
    """

    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __repr__(self):
        return f"[IntegerProxy name={self.name} value={self.value}]"

    def __eq__(self, other):
        other_value = other.value if isinstance(other, IntegerProxy) else other
        result = self.value == other_value

        Constraint(operator.eq, result, self, other)
        return result

    def __ne__(self, other):
        other_value = other.value if isinstance(other, IntegerProxy) else other
        result = self.value != other_value

        Constraint(operator.ne, result, self, other)
        return result

    def __le__(self, other):
        raise AssertionError("Not Implemented!")

    def __lt__(self, other):
        other_value = other.value if isinstance(other, IntegerProxy) else other
        result = self.value < other_value

        Constraint(operator.le, result, self, other)
        return result

    def __ge__(self, other):
        other_value = other.value if isinstance(other, IntegerProxy) else other
        result = self.value >= other_value

        Constraint(operator.ge, result, self, other)
        return result

    def __gt__(self, other):
        raise AssertionError("Not Implemented!")


# TODO: want this to pass isinstance(p, torch.Tensor) and isinstance(p, np.array) depending on
#   language context
# TODO: add method resolution through language context
class TensorProxy(object):
    """
    A wrapper for a tensor that records data and metadata accesses.

    TODO: implement all tensor metadata and data access methods
    TODO: consider delaying/avoiding string construction of names (possibly using ints for names when not debugging)
    """

    @staticmethod
    def _make_shape(name, shape):
        def _helper(idx, x):
            if isinstance(x, IntegerProxy):
                return x

            # Assumes x is an integer
            ip = IntegerProxy(f"{name}_{idx}", value=x)
            return ip

        my_shape = tuple(_helper(idx, x) for idx, x in enumerate(shape))
        return my_shape

    def __init__(self, name=None, *, shape, dtype):
        name = name if name is not None else get_trace().tensor_name()
        self.name = name

        # Shape is a tuple of integer proxies
        self.shape = self._make_shape(name, shape)

        # TODO: should ndim be an integer proxy, too?
        self.ndim = len(shape)

        self.dtype = dtype

    def __repr__(self):
        return f"[TensorProxy, name={self.name}, shape={self.shape}]"


# TODO: differentiate tensor and other types of names
def proxy(x):
    """
    Creates a proxy object.
    """

    # TODO: make this conditional on PyTorch being available
    # TODO: combine with TensorProxy case below
    if isinstance(x, torch.Tensor):
        name = get_trace().tensor_name()
        return TensorProxy(name, shape=x.shape, dtype=x.dtype)
    if isinstance(x, TensorProxy):
        name = get_trace().tensor_name()
        return TensorProxy(name, shape=x.shape, dtype=x.dtype)
    if isinstance(x, IntegerProxy):
        name = get_trace().tensor_name()
        return IntegerProxy()

    raise AssertionError(f"Can't proxy unknown type {type(x)}")
