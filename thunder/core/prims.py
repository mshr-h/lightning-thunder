from enum import Enum, auto
import operator
from numbers import Number
import builtins

from .trace import get_trace
from .proxies import NumberProxy, IntegerProxy, TensorProxy, proxy
import thunder.core.utils as utils
from .utils import (
    check,
    check_same_dtype,
    same_shape,
    dtype_to_type,
    get_numberlike_type,
    get_numberlike_value,
)

# This file defines Thunder's "primitive" operations. These are the
#   "building blocks" for all of Thunder's operators.

# Transforms and analysis defined on the primitive operations should
#   be inherited by the operation's they're composed of.

# This file depends on trace.py, proxies.py, and utils.py.

__all__ = [
    # Elementwise unary prims
    "abs",
    # Elementwise binary prims
    "add",
    "sub",
    # Shape prims
    "broadcast_in_dim",
    # Reduction prims
    "var",
]


class Ops(Enum):
    ABS = auto()
    ADD = auto()
    SUB = auto()
    BROADCAST_IN_DIM = auto()
    VAR = auto()


# maps from operators to their meta functions
# declared here but updated below
ops_to_meta_functions_map = {}
ops_to_pretty_name_map = {}

# Prim defintions
# A primitive definition needs to do the following:
#   - define the op's meta function
#       The meta function maps proxy inputs to a proxy output that has the same metadata
#       as the result of calling the operation with inputs that have the same metadata as the proxies.
#       Meta functions are called within a tracing context. TODO: relax this.
#   - call _make_prim

# TODO: add error context


class Prim(object):
    """
    A call to a primitive.

    Holds the inputs, outputs, and printing information.
    """

    # TODO: support returning multiple values
    def __init__(self, op, name, result, *args, **kwargs):
        self.op = op
        self.name = name
        self.result = result
        self.args = args
        self.kwargs = kwargs

    def __repr__(self):
        result_string = str(self.result)
        arg_string = ", ".join(str(arg) for arg in self.args)
        kwarg_string = ", ".join(f"{k}={v}" for k, v in self.kwargs.items())
        return f"[Prim {self.name}, \n\tresult=({result_string}), \n\targs=({arg_string}), \n\tkwargs={{{kwarg_string}}}]"


def _make_prim(id, meta, name, *, number_handler=None):
    ops_to_meta_functions_map[id] = meta
    ops_to_pretty_name_map[id] = name

    # TODO: update the signature
    # TODO: improve kwarg handling
    def _fn(*_args, **kwargs):
        t = get_trace()

        # Identifies number constants in args
        # TODO: review other types of constants
        def _extract_constant(x):
            if isinstance(x, Number) and not isinstance(x, NumberProxy):
                p = proxy(x)
                t.add_constant(p)
                return p
            return x

        args = tuple(map(_extract_constant, _args))

        result = meta(*args, **kwargs, name=name, number_handler=number_handler)
        sym = Prim(id, name, result, *args, **kwargs)
        t.add_symbol(sym)
        return result

    return _fn


#
# Elementwise unary prims
#


def _elementwise_unary_meta(a, *, name, number_handler=None, **kwargs):
    # Tensor case
    if isinstance(a, TensorProxy):
        return TensorProxy(tensor=a)

    # Number case
    check(
        isinstance(a, Number),
        lambda: f"Elementwise unary primitives don't support inputs of type {type(a)}!",
    )

    check(
        number_handler is not None,
        lambda: f"The elementwise unary primitive {name} doesn't support number inputs!",
    )

    a_typ = get_numberlike_type(a)
    va = get_numberlike_value(a)
    value = number_handler(va)
    return proxy(value)


abs = _make_prim(Ops.ABS, _elementwise_unary_meta, "abs", number_handler=builtins.abs)

#
# Elementwise binary prims
#

# TODO: add type promotion (ex. abs complex->float type promotion)
# TODO: document elementwise binary meta, incl. stride logic
def _elementwise_binary_meta(a, b, *, name, number_handler=None, **kwargs):

    # Tensors or Number inputs only
    if not isinstance(a, (TensorProxy, Number)):
        raise ValueError(f"Unexpected type {type(a)}!")
    if not isinstance(b, (TensorProxy, Number)):
        raise ValueError(f"Unexpected type {type(b)}!")

    # tensor x tensor case
    if isinstance(a, TensorProxy) and isinstance(b, TensorProxy):
        check(
            same_shape(a.shape, b.shape),
            lambda: f"Elementwise binary primitives require the shapes of the inputs tensors to be the same! But got shapes {a.shape} and {b.shape}!",
        )
        check(
            a.dtype == b.dtype,
            lambda: f"Elementwise binary primitives require the dtypes of the inputs tensors to be the same! But got dtypes {a.dtype} and {b.dtype}!",
        )
        return TensorProxy(tensor=a)

    # scalar x scalar case
    if isinstance(a, Number) and isinstance(b, Number):
        check(
            number_handler is not None,
            lambda: f"The elementwise binary primitive {name} doesn't support number x number inputs!",
        )

        a_typ = get_numberlike_type(a)
        b_typ = get_numberlike_type(b)
        check(
            a_typ is b_typ,
            lambda: f"Elementwise binary primitives require the types of numbers to be the same! But got types {a.typ} and {b.typ}!",
        )

        va, vb = get_numberlike_value(a), get_numberlike_value(b)
        value = number_handler(va, vb)
        return proxy(value)

    # tensor x scalar case
    tensor = a if isinstance(a, TensorProxy) else b
    number = b if tensor is a else a

    return TensorProxy(tensor=tensor)


add = _make_prim(Ops.ADD, _elementwise_binary_meta, "add", number_handler=operator.add)

sub = _make_prim(Ops.SUB, _elementwise_binary_meta, "sub", number_handler=operator.sub)

#
# Shape prims
#


def broadcast_in_dim_meta(a, shape, broadcast_dimensions, **kwargs):
    return TensorProxy(shape=shape, dtype=a.dtype)


broadcast_in_dim = _make_prim(
    Ops.BROADCAST_IN_DIM, broadcast_in_dim_meta, "broadcast_in_dim"
)

#
# Reduction prims
#


def _compute_reduction_output_shape(shape, dims):
    for idx in dims:
        utils.validate_idx(len(shape), idx)

    new_shape = []
    for idx in range(len(shape)):
        if idx in dims:
            continue

        new_shape.append(shape[idx])

    return tuple(new_shape)


def _reduction_meta(a, dims, *, output_dtype=None):
    """
    Meta function for single output reduction operations
    """

    if output_dtype is None:
        output_dtype = a.dtype

    output_shape = _compute_reduction_output_shape(a.shape, dims)

    return TensorProxy(
        shape=output_shape,
        dtype=output_dtype,
    )


def var_meta(a, dims, *, correction, **kwargs):
    if utils.is_complex_dtype(a.dtype):
        output_dtype = utils.corresponding_real_dtype(a.dtype)
    else:
        output_dtype = a.dtype
    return _reduction_meta(a, dims, output_dtype=output_dtype)


var = _make_prim(Ops.VAR, var_meta, "var")
