from enum import Enum, auto

from .trace import get_trace
from .proxies import TensorProxy, proxy
from .utils import check, check_same_dtype, same_shape

# This file defines Thunder's "primitive" operations. These are the
#   "building blocks" for all of Thunder's operators.

# Transforms and analysis defined on the primitive operations should
#   be inherited by the operation's they're composed of.

# This file depends on trace.py, proxies.py, and utils.py.

__all__ = [
    # Elementwise binary operations
    "add",
    # Shape operations
    "broadcast_in_dim",
]


class Ops(Enum):
    ADD = auto()
    BROADCAST_IN_DIM = auto()


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
        kwarg_string = ""
        return f"[Prim {self.name}, \n\tresult=({result_string}), \n\targs=({arg_string}), \n\tkwargs={{{kwarg_string}}}]"


def _make_prim(id, meta, name):
    ops_to_meta_functions_map[id] = meta
    ops_to_pretty_name_map[id] = name

    # TODO: update the signature
    def _fn(*args, **kwargs):
        t = get_trace()
        result = meta(*args, **kwargs)
        sym = Prim(id, name, result, *args, **kwargs)
        t.add_symbol(sym)
        return result

    return _fn


#
# Elementwise binary operations
#

# TODO: add type promotion
# TODO: add scalar support
# TODO: document elementwise binary meta, incl. stride logic
def _elementwise_binary_meta(a, b):
    # TODO: improve type checks

    if not isinstance(a, TensorProxy):
        raise ValueError(f"Unexpected type {type(a)}!")
    if not isinstance(b, TensorProxy):
        raise ValueError(f"Unexpected type {type(b)}!")

    # TODO: improve error messages
    check(
        same_shape(a.shape, b.shape),
        lambda: f"Elementwise binary primitives require the shapes of the tensors to be the same!",
    )
    check_same_dtype(a.dtype, b.dtype)

    return TensorProxy(shape=a.shape, dtype=a.dtype)


add = _make_prim(Ops.ADD, _elementwise_binary_meta, "add")

#
# Shape operations
#

def broadcast_in_dim_meta(a, shape, broadcast_dimensions):
    return TensorProxy(shape=shape, dtype=a.dtype)


broadcast_in_dim = _make_prim(
    Ops.BROADCAST_IN_DIM, broadcast_in_dim_meta, "broadcast_in_dim"
)
