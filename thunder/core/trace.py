# import contextvars  # TODO: review this (vs threadlocal?) -- currently used to set the current trace
import string
from collections import deque
from contextvars import ContextVar

# This file defines the tracing context, methods for acquiring it, and related classes.
# This file is the base of Thunder's file hierarchy. It is always safe to import.
# In the future, the tracing context will be elaborated on with more structure, like
#   regions and transitions between regions.
# Currently, traces are represented as single regions with no transitions. All their
#   constraints are evaluated before entry, and not incrementally as transitions
#   between distinct traces.

__all__ = [
    "Constraint",
    "Trace",
    "new_trace",
    "get_trace",
    "reset_trace",
    "set_language_context",
    "get_language_context",
    "reset_language_context",
    "set_executor_context",
    "get_executor_context",
    "reset_executor_context",
]

#
# ContextVars
#

# Holds the current Trace object
_trace = ContextVar("trace")


def new_trace():
    """Sets the current trace."""

    return _trace.set(Trace())


# TODO: add ability to get a temporary "anonymous" trace
# TODO: possibly add a kwarg to control this behavior
def get_trace():
    """Gets the current trace, returning None if there is no current trace."""

    try:
        t = _trace.get()
        return t
    except LookupError:
        pass

    return None


def reset_trace(token):
    """Resets the tracing state."""

    _trace.set(token)


# Holds the current language context
# TODO: unused
# NOTE: this file does not depend on the definition of the language context,
#   so it's an opaque object from the perspective of this file
_language_ctx = ContextVar("language_ctx")


def set_language_context(ctx):
    """Sets the current trace."""

    return _language_ctx.set(ctx)


# TODO: add ability to get a temporary "anonymous" trace
# TODO: possibly add a kwarg to control this behavior
def get_language_context():
    """Gets the current trace, returning None if there is no current trace."""

    try:
        ctx = _language_ctx.get()
        return ctx
    except LookupError:
        pass

    return ctx


def reset_language_context(token):
    """Resets the tracing state."""

    _language_ctx.set(token)


# Holds the current execution context
# NOTE: this file does not depend on the definition of the execution context,
#   so it's an opaque object from the perspective of this file
_executor_ctx = ContextVar("executor_ctx")


def set_executor_context(ctx):
    """Sets the current trace."""

    return _executor_ctx.set(ctx)


# TODO: add ability to get a temporary "anonymous" trace
# TODO: possibly add a kwarg to control this behavior
def get_executor_context():
    """Gets the current execution context, returning None if there is no current trace."""

    try:
        ctx = _executor_ctx.get()
        return ctx
    except LookupError:
        pass

    return ctx


def reset_executor_context(token):
    """Resets the tracing state."""

    _executor_ctx.set(token)


class Constraint:
    """Represents a "constraint" on the validity of a region.

    Constrains include a function to be evaluated, the arguments to it, and
    the expected result.

    For example, if an integer proxy (see proxies.py) was compared with 5 (p == 5), and p
    did equal 5, then that would create a constraint (==, args=(p, 5), expected=True).
    """

    def __init__(self, op, expected, *args):
        self.op = op
        self.expected = expected
        self.args = args

        t = get_trace()
        t.add_constraint(self)

    def __repr__(self):
        return f"[Constraint op={str(self.op)} expected={str(self.expected)} args={[str(arg) for arg in self.args]}]"


class Trace:
    """The tracing context.

    Contains common datastructures to track the trace.
    """

    def __init__(self):
        self.inputs = deque()
        self.kwargs = deque()
        self.constants = deque()
        self.outputs = deque()
        self.symbols = deque()
        self.constraints = deque()

        self._lowercase = tuple(string.ascii_lowercase)
        self._tensor_name_counter = 0

        self._uppercase = tuple(string.ascii_uppercase)
        self._constant_name_counter = 0

    def __repr__(self):
        constraint_string = "\n".join(str(constraint) for constraint in self.constraints)
        input_string = "\n".join(str(inp) for inp in self.inputs)
        kwarg_input_string = "\n".join(f"{k}={v}" for k, v in self.kwargs)
        constant_string = "\n".join(str(constant) for constant in self.constants)
        symbol_string = "\n".join(str(sym) for sym in self.symbols)
        output_string = "\n".join(str(out) for out in self.outputs)
        return (
            f"[Trace, \nConstraints:\n{constraint_string}\nInputs:\n{input_string}\n"
            f"Kwarg Inputs:\n{kwarg_input_string}\nConstants:\n{constant_string}\n"
            f"Symbols:\n{symbol_string}\nOutputs:\n{output_string}]"
        )

    # TODO: Consider using a different name generation technique that reuses the original names
    @staticmethod
    def _create_name(idx, chars):
        name = ""
        while idx >= len(chars):
            name = chars[idx % len(chars)] + name
            idx = idx - len(chars)
        name = chars[idx] + name

        return name

    # TODO: add name functions for other types, like integers
    # TODO: expose common helpers like this so that callers don't need to acquire the trace
    #   the trace context itself should be invisible to them
    def tensor_name(self):
        idx = self._tensor_name_counter
        self._tensor_name_counter += 1

        return self._create_name(idx, self._lowercase)

    def constant_name(self):
        idx = self._constant_name_counter
        self._constant_name_counter += 1

        return self._create_name(idx, self._uppercase)

    def add_input(self, inp):
        self.inputs.append(inp)
        return inp

    def add_kwarg_input(self, k, v):
        self.kwargs.append((k, v))
        return k, v

    # TODO: review constants, particularly Python lists
    def add_constant(self, constant):
        self.constants.append(constant)
        return constant

    def add_output(self, out):
        self.outputs.append(out)
        return out

    def add_symbol(self, sym):
        self.symbols.append(sym)
        return sym

    def add_constraint(self, constraint):
        self.constraints.append(constraint)
        return constraint
