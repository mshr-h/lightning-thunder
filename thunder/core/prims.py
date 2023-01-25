import builtins
import operator
from enum import auto, Enum
from functools import partial
from numbers import Number
import math


import thunder.core.utils as utils
import thunder.core.dtypes as dtypes
from .proxies import NumberProxy, proxy, TensorProxy
from .trace import get_trace
from .utils import check, get_numberlike_value, same_shape

# This file defines Thunder's "primitive" operations. These are the
#   "building blocks" for all of Thunder's operators.

# Transforms and analysis defined on the primitive operations should
#   be inherited by the operation's they're composed of.

# This file depends on trace.py, the dtypes submodule, proxies.py, and utils.py.

__all__ = [
    # Methods and datastructures for constructing primitive operations
    "make_prim",
    # Data movement and transformation prims
    "convert_element_type",
    # Tensor creation prims
    "full",
    # Shape prims
    "broadcast_in_dim_meta",
    "broadcast_in_dim",
    # Elementwise unary prims
    "abs",
    "acos",
    "acosh",
    "asin",
    "atan",
    "atanh",
    "bitwise_not",
    "ceil",
    "cos",
    "cosh",
    "erf",
    "erfc",
    "exp",
    "expm1",
    "floor",
    "isfinite",
    "tanh",
    # Elementwise binary prims
    "add",
    "atan2",
    "bitwise_and",
    "div",
    "mul",
    "pow",
    "sub",
    # Reduction prims
    "reduction_meta",
    "sum_meta",
    "sum",
    "var",
    "var_meta",
    # Matmul prims
    "linear",
]


class Ops(Enum):
    # Data movement and transformation prims
    CONVERT_ELEMENT_TYPE = auto()
    # Tensor creation prims
    FULL = auto()
    # Elementwise unary prims
    ABS = auto()
    ACOS = auto()
    ACOSH = auto()
    ASIN = auto()
    ATAN = auto()
    ATANH = auto()
    BITWISE_NOT = auto()
    CEIL = auto()
    COS = auto()
    COSH = auto()
    ERF = auto()
    ERFC = auto()
    EXP = auto()
    EXPM1 = auto()
    FLOOR = auto()
    ISFINITE = auto()
    TANH = auto()
    # Elementwise binary prims
    ADD = auto()
    ATAN2 = auto()
    BITWISE_AND = auto()
    DIV = auto()
    MUL = auto()
    POW = auto()
    SUB = auto()
    # Shape prims
    BROADCAST_IN_DIM = auto()
    # Reduction prims
    SUM = auto()
    VAR = auto()
    # Matmul prims
    LINEAR = auto()


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
#   - call make_prim

# TODO: add error context


class Prim:
    """A call to a primitive.

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
        return (
            f"[Prim {self.name}, \n\tresult=({result_string}), \n\targs=({arg_string}), \n\tkwargs={{{kwarg_string}}}]"
        )


def make_prim(id, name, meta):
    # TODO: probably want to consolidate these maps by having one map
    #   to a prim data object with these attributes
    #   (or possibly to a Prim class and rename the class that is inserted into traces)
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
# Data movement and transformation prims
#


# TODO: consider supporting number subclasses
def _convert_element_type_meta(a, dtype):
    if isinstance(a, Number):
        utils.check(utils.is_numbertype(dtype), lambda: f"Trying to convert a number to non-numbertype object {dtype}!")
        result = dtype(utils.get_numberlike_value(a))
        proxy_name = get_trace().make_proxy_name()
        return proxy(result, name=proxy_name)

    # a is a Tensor
    proxy_name = get_trace().make_proxy_name()
    return TensorProxy(name=proxy_name, tensor=a, dtype=dtype)


convert_element_type = make_prim(Ops.CONVERT_ELEMENT_TYPE, "convert_element_type", _convert_element_type_meta)

#
# Tensor creation prims
#


# TODO: add some architecture for constructing tensor creation prims
# TODO: add device support to tensor proxies
def _full_meta(shape, fill_value, *, dtype, device):
    proxy_name = get_trace().make_proxy_name()
    return TensorProxy(name=proxy_name, shape=shape, device=device, dtype=dtype)


full = make_prim(Ops.FULL, "full", _full_meta)

#
# Elementwise prims
#

# Describes how an elementwise primitive type promotes.
# NOTE: this is distinct from ELEMENTWISE_TYPE_PROMOTION_KIND in utils.py,
#   which describes how user-facing elementwise operations type promote.
# This type promotion just maps an input type to a result type.
# DEFAULT means the result type is the same as the input type.
# ALWAYS_BOOL means the result type is always bool.
# COMPLEX_TO_FLOAT means the result type is determined like for DEFAULT, unless
#   the input type is complex, in which case the result type is the corresponding
#   float type.
# Examples uses:
#  - DEFAULT: add
#  - ALWAYS_BOOL: isfinite
#  - COMPLEX_TO_FLOAT: abs


class ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND(Enum):
    DEFAULT = auto()
    ALWAYS_BOOL = auto()
    COMPLEX_TO_FLOAT = auto()


def _prim_type_promotion(typ, type_promotion_kind):
    if type_promotion_kind is ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT:
        return typ

    if type_promotion_kind is ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.ALWAYS_BOOL:
        return bool

    if type_promotion_kind is ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.COMPLEX_TO_FLOAT:
        if utils.is_complex_dtype(typ):
            return utils.corresponding_real_dtype(typ)

        return typ

    raise AssertionError("Unknown prim type promotion kind {type_promotion_kind}!")


#
# Elementwise unary prims
#

# Elementwise unary prims to implement:
# "asinh",
# "cbrt",
# "digamma",
# "erf_inv",
# "erfcx",
# "exp2",
# "fill",
# "imag",
# "lgamma",
# "log",
# "log1p",
# "log2",
# "log10",
# "ndtri",
# "neg",
# "real",
# "reciprocal",
# "round",
# "sign",
# "signbit",
# "sin",
# "sinh",
# "sqrt",
# "tan",
# "tanh",
# "trunc",

# nvFuser unary ops (from https://github.com/pytorch/pytorch/blob/master/torch/_prims/nvfuser_prims.py)
# "imag",
# "lgamma",
# "log",
# "log1p",
# "log2",
# "log10",
# "reciprocal",
# "neg",
# "real",
# "round",
# "rsqrt",
# "sign",
# "sin",
# "sinh",
# "sqrt",
# "tan",
# "tanh",
# "trunc",

# TODO: review number handlers for complex support


def _elementwise_unary_meta(a, *, name, type_promotion_kind, number_handler=None, **kwargs):
    # TODO: break fn into two, one for returning types, one for checking for equality?
    input_dtype = utils.to_dtype(a, true_dtype=True)

    result_dtype = _prim_type_promotion(input_dtype, type_promotion_kind=type_promotion_kind)
    proxy_name = get_trace().make_proxy_name()

    # Tensor case
    if isinstance(a, TensorProxy):
        return TensorProxy(name=proxy_name, tensor=a, dtype=result_dtype)

    # Number case
    check(
        isinstance(a, Number),
        lambda: f"Elementwise unary primitives don't support inputs of type {type(a)}!",
    )

    check(
        number_handler is not None,
        lambda: f"The elementwise unary primitive {name} doesn't support number inputs!",
    )

    # a_typ = get_numberlike_type(a)
    va = get_numberlike_value(a)
    result = result_dtype(number_handler(va))
    return proxy(result, name=proxy_name)


abs = make_prim(
    Ops.ABS,
    "abs",
    partial(
        _elementwise_unary_meta,
        name="abs",
        type_promotion_kind=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.COMPLEX_TO_FLOAT,
        number_handler=builtins.abs,
    ),
)

acos = make_prim(
    Ops.ACOS,
    "acos",
    partial(
        _elementwise_unary_meta,
        name="acos",
        type_promotion_kind=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
        number_handler=math.acos,
    ),
)

acosh = make_prim(
    Ops.ACOSH,
    "acosh",
    partial(
        _elementwise_unary_meta,
        name="acosh",
        type_promotion_kind=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
        number_handler=math.acosh,
    ),
)

asin = make_prim(
    Ops.ASIN,
    "asin",
    partial(
        _elementwise_unary_meta,
        name="asin",
        type_promotion_kind=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
        number_handler=math.asin,
    ),
)

atan = make_prim(
    Ops.ATAN,
    "atan",
    partial(
        _elementwise_unary_meta,
        name="atan",
        type_promotion_kind=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
        number_handler=math.atan,
    ),
)

atanh = make_prim(
    Ops.ATANH,
    "atanh",
    partial(
        _elementwise_unary_meta,
        name="atanh",
        type_promotion_kind=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
        number_handler=math.atanh,
    ),
)

bitwise_not = make_prim(
    Ops.BITWISE_NOT,
    "bitwise_not",
    partial(
        _elementwise_unary_meta,
        name="bitwise_not",
        type_promotion_kind=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
        number_handler=operator.invert,
    ),
)

ceil = make_prim(
    Ops.CEIL,
    "ceil",
    partial(
        _elementwise_unary_meta,
        name="ceil",
        type_promotion_kind=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
        number_handler=math.ceil,
    ),
)

cos = make_prim(
    Ops.COS,
    "cos",
    partial(
        _elementwise_unary_meta,
        name="cos",
        type_promotion_kind=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
        number_handler=math.cos,
    ),
)

cosh = make_prim(
    Ops.COSH,
    "cosh",
    partial(
        _elementwise_unary_meta,
        name="cosh",
        type_promotion_kind=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
        number_handler=math.cosh,
    ),
)

erf = make_prim(
    Ops.ERF,
    "erf",
    partial(
        _elementwise_unary_meta,
        name="erf",
        type_promotion_kind=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
        number_handler=math.erf,
    ),
)

erfc = make_prim(
    Ops.ERFC,
    "erfc",
    partial(
        _elementwise_unary_meta,
        name="erfc",
        type_promotion_kind=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
        number_handler=math.erfc,
    ),
)

exp = make_prim(
    Ops.EXP,
    "exp",
    partial(
        _elementwise_unary_meta,
        name="exp",
        type_promotion_kind=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
        number_handler=math.exp,
    ),
)

expm1 = make_prim(
    Ops.EXPM1,
    "expm1",
    partial(
        _elementwise_unary_meta,
        name="expm1",
        type_promotion_kind=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
        number_handler=math.expm1,
    ),
)

floor = make_prim(
    Ops.FLOOR,
    "floor",
    partial(
        _elementwise_unary_meta,
        name="floor",
        type_promotion_kind=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
        number_handler=math.floor,
    ),
)

isfinite = make_prim(
    Ops.ISFINITE,
    "isfinite",
    partial(
        _elementwise_unary_meta,
        name="isfinite",
        type_promotion_kind=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
        number_handler=math.isfinite,
    ),
)

tanh = make_prim(
    Ops.TANH,
    "tanh",
    partial(
        _elementwise_unary_meta,
        name="isfinite",
        type_promotion_kind=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
        number_handler=math.tanh,
    ),
)


#
# Elementwise binary prims
#
# "bitwise_or",
# "bitwise_xor",
# # 'complex',  # needs custom meta
# "eq",
# "fmax",
# "fmin",
# "fmod",
# "gcd",
# "ge",
# "gt",
# "hypot",
# "igamma",
# "igammac",
# "le",
# "lt",
# "maximum",
# "minimum",
# "ne",
# "nextafter",
# "remainder",
# "rsqrt",
# "shift_left",
# "shift_right_arithmetic",
# "shift_right_logical",  # not implemented
# "zeta",

# nvFuser binary ops (from https://github.com/pytorch/pytorch/blob/master/torch/_prims/nvfuser_prims.py)
# "bitwise_or",
# "bitwise_xor",
# "eq",
# "fmod",
# "ge",
# "gt",
# "le",
# "lt",
# "ne",
# "remainder",


# TODO: add type promotion (ex. abs complex->float type promotion)
# TODO: document elementwise binary meta, incl. stride logic
# TODO: use supported_dtypes
# TODO: correct name of output
def _elementwise_binary_meta(
    a, b, *, name, type_promotion_kind, number_handler=None, supported_dtypes=(dtypes.dtype,), **kwargs
):
    # Tensors or Number inputs only
    if not isinstance(a, (TensorProxy, Number)):
        raise ValueError(f"Unexpected type {type(a)}!")
    if not isinstance(b, (TensorProxy, Number)):
        raise ValueError(f"Unexpected type {type(b)}!")

    # Inputs must have the same dtype
    numbertype, dtype = utils.check_same_dtype(a, b)
    input_type = dtype if dtype is not None else numbertype

    result_type = _prim_type_promotion(input_type, type_promotion_kind=type_promotion_kind)
    proxy_name = get_trace().make_proxy_name()

    # tensor x tensor case
    if isinstance(a, TensorProxy) and isinstance(b, TensorProxy):
        check(
            same_shape(a.shape, b.shape),
            lambda: (
                "Elementwise binary primitives require the shapes of the inputs tensors to "
                f"be the same! But got shapes {a.shape} and {b.shape}!"
            ),
        )

        return TensorProxy(name=proxy_name, tensor=a, dtype=result_type)

    # scalar x scalar case
    if isinstance(a, Number) and isinstance(b, Number):
        check(
            number_handler is not None,
            lambda: f"The elementwise binary primitive {name} doesn't support number x number inputs!",
        )

        va, vb = get_numberlike_value(a), get_numberlike_value(b)
        value = number_handler(va, vb)
        result = result_type(value)
        return proxy(result, name=proxy_name)

    # tensor x scalar case
    tensor = a if isinstance(a, TensorProxy) else b
    # number = b if tensor is a else a

    return TensorProxy(name=proxy_name, tensor=tensor, dtype=result_type)


add = make_prim(
    Ops.ADD,
    "add",
    partial(
        _elementwise_binary_meta,
        name="add",
        type_promotion_kind=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
        number_handler=operator.add,
    ),
)

atan2 = make_prim(
    Ops.ATAN2,
    "atan2",
    partial(
        _elementwise_binary_meta,
        name="atan2",
        type_promotion_kind=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
    ),
)

bitwise_and = make_prim(
    Ops.BITWISE_AND,
    "bitwise_and",
    partial(
        _elementwise_binary_meta,
        name="bitwise_and",
        type_promotion_kind=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
        supported_dtypes=(dtypes.exact,),
    ),
)


def _div_number_handler(a, b):
    if isinstance(a, (float, complex)):
        return a / b

    # int (and bool) case, performs floor division
    return a // b


div = make_prim(
    Ops.DIV,
    "div",
    partial(
        _elementwise_binary_meta,
        name="div",
        type_promotion_kind=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
        number_handler=_div_number_handler,
    ),
)

mul = make_prim(
    Ops.MUL,
    "mul",
    partial(
        _elementwise_binary_meta,
        name="mul",
        type_promotion_kind=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
        number_handler=operator.mul,
    ),
)

pow = make_prim(
    Ops.POW,
    "pow",
    partial(
        _elementwise_binary_meta,
        name="pow",
        type_promotion_kind=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
        number_handler=operator.pow,
    ),
)

sub = make_prim(
    Ops.SUB,
    "sub",
    partial(
        _elementwise_binary_meta,
        name="sub",
        type_promotion_kind=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
        number_handler=operator.sub,
    ),
)

#
# Shape prims
#


def broadcast_in_dim_meta(a, shape, broadcast_dimensions, **kwargs):
    proxy_name = get_trace().make_proxy_name()
    return TensorProxy(name=proxy_name, shape=shape, device=a.device, dtype=a.true_dtype)


broadcast_in_dim = make_prim(
    Ops.BROADCAST_IN_DIM,
    "broadcast_in_dim",
    broadcast_in_dim_meta,
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


def reduction_meta(a, dims, *, output_dtype=None, **kwargs):
    """Meta function for single output reduction operations."""

    if output_dtype is None:
        output_dtype = a.true_dtype

    output_shape = _compute_reduction_output_shape(a.shape, dims)

    proxy_name = get_trace().make_proxy_name()
    return TensorProxy(
        name=proxy_name,
        shape=output_shape,
        device=a.device,
        dtype=output_dtype,
    )


sum_meta = reduction_meta
sum = make_prim(Ops.SUM, "sum", sum_meta)


def var_meta(a, dims, *, correction, **kwargs):
    output_dtype = None
    if utils.is_complex_dtype(a.dtype):
        output_dtype = utils.corresponding_real_dtype(a.true_dtype)
    else:
        output_dtype = a.true_dtype
    return reduction_meta(a, dims, output_dtype=output_dtype)


var = make_prim(Ops.VAR, "var", var_meta)

#
# Matmul prims
#
# NOTE: matmul prims are highly experimental and will almost definitely change

# out = a @ w.transpose() + bias
def linear_meta(a, w, bias):
    # a's shape is (batch dims..., in)
    # w's shape is (out x in)
    # if bias is not None, bias's shape is (out)
    # the output shape is (batch dims..., out)

    # Checks types of the required arguments
    utils.check(isinstance(a, TensorProxy), lambda: f"a={a} was not a TensorProxy!")
    utils.check(isinstance(w, TensorProxy), lambda: f"w={w} was not a TensorProxy!")

    # Checks that required arguments are on the same device
    utils.check(a.device == w.device, lambda: f"Expected a.device={a.device} and w.device={w.device} to be the same!")

    # Acquires the computation dtype and checks that a and w have the same dtype
    dtype = a.dtype
    utils.check(
        dtypes.are_same_dtypes(a, w), lambda: f"Expected a.dtype={a.dtype} and w.dtype={w.dtype} to be the same!"
    )

    # Acquires the shape information and validates the shapes of the required arguments
    batch_dims = a.shape[:-1]
    in_length = a.shape[-1]

    # Validates w's shape
    utils.check(
        len(w.shape) == 2, lambda: f"Expected w.shape={w.shape} to have length 2, but found length {len(w.shape)}!"
    )
    utils.check(
        w.shape[1] == in_length,
        lambda: f"Expected w.shape={w.shape} to have an innermost dimension of length {in_length}, the same length as the innermost dimension of a.shape={a.shape}!",
    )

    out_length = w.shape[0]

    # Validates bias shape
    if bias is not None:
        utils.check(isinstance(bias, TensorProxy), lambda: f"bias={bias} was not None or a TensorProxy!")
        utils.check(
            a.device == bias.device,
            lambda: f"Expected a.device={a.device} and bias.device={bias.device} to be the same!",
        )
        utils.check(
            len(bias.shape) == 1,
            lambda: f"Expected bias.shape={bias.shape} to have length 1, but found length {len(bias.shape)}!",
        )
        utils.check(
            bias.shape[0] == out_length,
            lambda: f"Expected bias.shape={bias.shape} to have an innermost dimension of length {out_length}, the same length as the outermost dimension of w.shape={w.shape}!",
        )
        utils.check(
            dtypes.are_same_dtypes(bias, a),
            lambda: f"Expected a.dtype={a.dtype} and bias.dtype={bias.dtype} to be the same!",
        )

    out_shape = batch_dims + (out_length,)
    proxy_name = get_trace().make_proxy_name()
    return TensorProxy(name=proxy_name, shape=out_shape, device=a.device, dtype=dtype)


linear = make_prim(Ops.LINEAR, "linear", linear_meta)
