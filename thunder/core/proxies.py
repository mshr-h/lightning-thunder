import operator
from collections import deque
from numbers import Number
from functools import partial
from enum import auto, Enum

import thunder.core.dtypes as dtypes

from .trace import Constraint, get_language_context, get_trace

# This file defines Thunder's most basic proxies, stand-ins for other Python objects that
#   record Python interactions for the tracing context.

# This file depends on trace.py and the dtypes submodule.

__all__ = [
    # Number proxies
    "Proxy",
    "NumberProxy",
    "IntegerProxy",
    # Tensor proxy
    "TensorProxy",
    # Proxy helpers and types
    "proxy",
    # "NumberLike",
]


class Proxy:
    pass


class NumberProxy(Proxy):
    def __init__(self, python_type, name, value):
        self.python_type = python_type
        self.name = name
        self.value = value


# NOTE: Why no bool proxy? Because bool cannot be subclassed. There are no bool
#   instances, just True and False. Further, isinstance(True, int) is True in Python!
#   So bools get handled by IntegerProxy.

# TODO: implement more methods
#   See https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types
class IntegerProxy(NumberProxy, int):
    """A wrapper for an integer that records the dunder methods called on it and their results as constraints.

    TODO: implement all dunder methods
    """

    def __new__(cls, value, name=None):
        return int.__new__(cls, value)

    def __init__(self, value, name=None):
        # TODO: update to call a number name function
        name = name if name is not None else get_trace().constant_name()

        # NOTE: bools are also integers in Python
        python_type = bool if isinstance(value, bool) else int
        NumberProxy.__init__(self, python_type, name, value)

    def __repr__(self):
        return f"[IntegerProxy name={self.name} value={self.value}]"

    def __hash__(self):
        return super().__hash__()

    # NOTE: it'd be nice to define the following dunders to preserve proxies
    #   across calls to int() and float(), but returning "strict subclasses" of
    #   float is deprecated.

    # __int__

    # __float__

    def int(self):
        return self

    def float(self):
        return FloatProxy(float(self.value))

    def __add__(self, other):
        ctx = get_language_context()
        return ctx.add(self, other)

    def __sub__(self, other):
        ctx = get_language_context()
        return ctx.sub(self, other)

    def __truediv__(self, other):
        ctx = get_language_context()
        return ctx.true_divide(self, other)

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


class FloatProxy(NumberProxy, float):
    def __new__(cls, value, name=None):
        return float.__new__(cls, value)

    def __init__(self, value, name=None):
        # TODO: update to call a number name function
        name = name if name is not None else get_trace().constant_name()
        NumberProxy.__init__(self, float, name, value)

    def __repr__(self):
        return f"[FloatProxy name={self.name} value={self.value}]"

    # NOTE: it'd be nice to define the following dunders to preserve proxies
    #   across calls to int() and float(), but returning "strict subclasses" of
    #   float is deprecated.

    # __int__

    # __float__

    def int(self):
        return IntegerProxy(int(self.value))

    def float(self):
        return self

    def __add__(self, other):
        ctx = get_language_context()
        return ctx.add(self, other)

    def __sub__(self, other):
        ctx = get_language_context()
        return ctx.sub(self, other)

    def __truediv__(self, other):
        ctx = get_language_context()
        return ctx.true_divide(self, other)


class TensorMethods(Enum):
    ADD = auto()


# TODO: want this to pass isinstance(p, torch.Tensor) and isinstance(p, np.array) depending on
#   language context
# TODO: add method resolution through language context
class TensorProxy(Proxy):
    """A wrapper for a tensor that records data and metadata accesses.

    TODO: implement all tensor metadata and data access methods
    TODO: consider delaying/avoiding string construction of names (possibly using ints for names when not debugging)
    """

    @staticmethod
    def _make_shape(name, shape):
        def _helper(idx, x):
            if isinstance(x, IntegerProxy):
                return x

            # Assumes x is an integer
            ip = IntegerProxy(value=x, name=f"{name}_{idx}")
            return ip

        my_shape = tuple(_helper(idx, x) for idx, x in enumerate(shape))
        return my_shape

    def __init__(
        self,
        *,
        name=None,
        tensor=None,
        shape=None,
        device=None,
        dtype=None,
    ):
        name = name if name is not None else get_trace().tensor_name()
        self.name = name

        if dtype is not None:
            assert dtypes.is_dtype(dtype), f"Unknown valuetype={dtype}!"

        if tensor is not None:
            # Pulls metadata from the tensor, but explicit kwargs take precedence
            assert isinstance(tensor, TensorProxy)
            self.shape = tensor.shape if shape is None else self._make_shape(name, shape)
            self._dtype = tensor.dtype if dtype is None else dtype
            self.device = tensor.device if device is None else device
        else:
            # Requires all metadata be specified explicitly
            assert shape is not None
            assert device is not None
            assert dtype is not None

            # Shape is a tuple of integer proxies
            self.shape = self._make_shape(name, shape)
            self.device = device
            self._dtype = dtype

        # TODO: should ndim be an integer proxy, too?
        self.ndim = len(self.shape)

        # Canonicalizes numbertypes to datatypes
        if dtypes.is_numbertype(self._dtype):
            self._dtype = dtypes.numbertype_to_dtype(self._dtype)

    def __repr__(self):
        return f"[TensorProxy, name={self.name}, shape={self.shape}, dtype={self.dtype}, is_weak={dtypes.is_weak_dtype(self._dtype)}]"

    # .dtype, registered using __getattr__
    def _get_dtype(self):
        """
        Returns the strong variant of the tensor's dtype.

        To acquire the actual dtype use "true_dtype"
        """
        return dtypes.to_strong_dtype(self._dtype)

    # .true_dtype, registered using __getattr__
    def _get_true_dtype(self):
        return self._dtype

    # +
    def __add__(self, other):
        ctx = get_language_context()
        return ctx.add(self, other)

    # *
    def __mul__(self, other):
        ctx = get_language_context()
        return ctx.mul(self, other)

    # -
    def __sub__(self, other):
        ctx = get_language_context()
        return ctx.sub(self, other)

    # /
    def __truediv__(self, other):
        ctx = get_language_context()
        return ctx.true_divide(self, other)

    # NOTE: If an attribute wasn't found, this assumes the attribute is a method defined
    #  by the language context. Just returning that method wouldn't work, however,
    #  since the TensorProxy, passed as self here, wouldn't be passed through to the
    #  actual method. That's why this partials the returned method.
    def __getattr__(self, name):
        # Handles properties
        if name == "dtype":
            return self._get_dtype()
        if name == "true_dtype":
            return self._get_true_dtype()

        ctx = get_language_context()
        return partial(getattr(ctx, name), self)


# TODO: differentiate tensor and other types of names
def proxy(x):
    """Creates a proxy object."""

    if isinstance(x, TensorProxy):
        name = get_trace().tensor_name()
        return TensorProxy(name=name, shape=x.shape, device=str(x.device.type), dtype=x.true_dtype)
    if isinstance(x, NumberProxy):
        return x
    if isinstance(x, int):
        return IntegerProxy(value=x)
    if isinstance(x, float):
        return FloatProxy(value=x)

    raise ValueError(f"Can't proxy unknown type {type(x)}")
