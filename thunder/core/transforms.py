from contextlib import nullcontext
from enum import auto, Enum
from functools import lru_cache, partial
from typing import Any, Callable, Dict, Sequence

from .. import make_trace, make_traced
from ..executors.torch import ops_to_torch_ops_map
from . import prims
from .proxies import Proxy
from .trace import detached_trace, get_trace, Trace
from .utils import safe_map, safe_zip, unzip2


class Transforms(Enum):
    IdentityOp = auto()
    JvpOp = auto()


@lru_cache(maxsize=None)
def symbol_to_eval(symbol: prims.Symbol):
    """Map a symbol to a function that evaluates it.

    Args:
        symbol: symbol to map
    """
    meta_func = prims.ops_to_meta_functions_map[symbol.op]

    prim_func = getattr(prims, symbol.name, None)
    if prim_func is not None:
        return prim_func

    return prims.eval_meta_and_record_symbol_fn(meta_func, symbol.op, symbol.name)


def eval_trace(trace, *args, symbol_mapper=symbol_to_eval, **kwargs):
    """Evaluate a trace.

    Args:
        trace: trace to evaluate
        *args: arguments to evaluate the trace with
        symbol_mapper: function that maps a symbol to a function that evaluates it
        **kwargs: keyword arguments to evaluate the trace with

    Returns:
        result of evaluating the trace
    """
    env = {}

    def read(x: Proxy):
        if isinstance(x, Proxy):
            return env[x]
        else:
            return x

    def write(v: Proxy, val: Any) -> None:
        assert v not in env
        env[v] = val

    safe_map(write, trace.args, args)
    safe_map(write, trace.kwargs.values(), kwargs.values())

    for symbol in trace.symbols:
        args = safe_map(read, symbol.args)
        kwargs = {k: read(v) for k, v in symbol.kwargs.items()}
        prim_func = symbol_mapper(symbol)
        result = prim_func(*args, **kwargs)
        if not isinstance(result, Sequence):
            result = (result,)
        safe_map(write, symbol.outputs, result)

    if not isinstance(trace.outputs, Sequence):
        return read(trace.outputs)
    return safe_map(read, trace.outputs)


def _identity_call_metafunc(*args, trace: Trace, **kwargs):
    with detached_trace():
        return eval_trace(trace, *args, **kwargs)


identity_call = prims.make_prim(Transforms.IdentityOp, "identity_call", _identity_call_metafunc)


def identity(func):
    """Identity transform for a Thunder function.

    Args:
        func (Callable): A Thunder function to be transformed.
    """

    def wrapper(*args, **kwargs):
        trace = make_trace(func)(*args, **kwargs)
        return identity_call(*args, **kwargs, trace=trace)

    return wrapper


def _identity_call_pytorch(*args, trace: Trace, **kwargs):
    import torch

    def symbol_mapper(op):
        if op.op == Transforms.IdentityOp:
            return _identity_call_pytorch

        torch_op = ops_to_torch_ops_map[op.op]
        if isinstance(torch_op, str):
            return getattr(torch, torch_op.strip("pytorch."))
        return torch_op

    with detached_trace():
        return eval_trace(trace, *args, **kwargs, symbol_mapper=symbol_mapper)


# Register the identity call for PyTorch executor.
ops_to_torch_ops_map[Transforms.IdentityOp] = _identity_call_pytorch


# Inline transform
# ----------------
# The inline transform is a special case of the identity transform.
# It is used to inline the transformation of a function in the trace without
# removing separate transform primitives from the trace.
inline_transforms_map: Dict[prims.Symbol, Callable] = dict()


def inline_symbol_mapper(symbol: prims.Symbol):
    if symbol.op in inline_transforms_map:
        return inline_transforms_map[symbol.op]

    return symbol_to_eval(symbol)


def _identity_call_inline(*args, trace: Trace, **kwargs):
    return eval_trace(trace, *args, **kwargs, symbol_mapper=inline_symbol_mapper)


inline_transforms_map[Transforms.IdentityOp] = _identity_call_inline


def inline(func):
    """Inline transform for a Thunder function.

    Args:
        func (Callable): A Thunder function to be transformed.
    """

    def wrapper(*args, **kwargs):
        trace = make_trace(func)(*args, **kwargs)
        return eval_trace(trace, *args, **kwargs, symbol_mapper=inline_symbol_mapper)

    return wrapper


# JVP transform
# -------------


def sin_jvp(a, ȧ):
    return prims.sin(a), prims.cos(a) * ȧ


def mul_jvp(a, b, ȧ, ḃ):
    return a * b, a * ḃ + b * ȧ


def add_jvp(a, b, ȧ, ḃ):
    return a + b, ȧ + ḃ


jvp_impls: Dict[prims.Symbol, Callable] = dict()

jvp_impls[prims.Ops.SIN] = sin_jvp
jvp_impls[prims.Ops.MUL] = mul_jvp
jvp_impls[prims.Ops.ADD] = add_jvp


def jvp_symbol_mapper(symbol: prims.Symbol):
    """Maps a symbol to a JVP function that evaluates it.

    Args:
        symbol (prims.Symbol): Symbol to evaluate.

    Raises:
        NotImplementedError: If the JVP for the symbol is not implemented.

    Returns:
        Callable: JVP function that evaluates the symbol.
    """
    # NOTE: I'm sorry for this code, but it's the best I could do.
    # if symbol.args doesn't have subclasses of Proxy, then we need to return a zero tangent
    if not any(isinstance(arg, Proxy) for arg in symbol.args):

        def _jvp_impl_const(symbol, *args, **kwargs):
            # TODO: this is weird, but we need to return a tuple of tuples
            out = symbol_to_eval(symbol)(*args, **kwargs)
            if isinstance(out, Sequence):
                return ((out, tuple(0 for x in out)),)
            return ((out, 0),)

        return partial(_jvp_impl_const, symbol)

    # Normal case, we have a proxy tangent
    jvp_impl = jvp_impls.get(symbol.op)
    if jvp_impl is None:
        raise NotImplementedError(f"JVP for {symbol.op} is not implemented")

    def _jvp_impl(*args, **kwargs):
        primals, tangents = unzip2(args)
        return (jvp_impl(*primals, *tangents, **kwargs),)

    return _jvp_impl


def _jvp_call_metafunc(primals, tangents, trace: Trace, detached: bool, **kwargs):
    assert len(kwargs) == 0, "JVP for kwargs is not implemented"

    def jvp_func(*primals_and_tangents):
        _primals, _tangents = (
            primals_and_tangents[: len(primals)],
            primals_and_tangents[len(primals) :],
        )
        primals_tangents_pairs = safe_zip(_primals, _tangents)
        outs = eval_trace(trace, *primals_tangents_pairs, symbol_mapper=jvp_symbol_mapper)
        return outs

    ctx = detached_trace() if detached else nullcontext()
    with ctx:
        return jvp_func(*primals, *tangents)


jvp_call = prims.make_prim(Transforms.JvpOp, "jvp_call", partial(_jvp_call_metafunc, detached=True))
inline_transforms_map[Transforms.JvpOp] = partial(_jvp_call_metafunc, detached=False)


def _identity_call_jvp(*primals_and_tangents, trace: Trace, **kwargs):
    half = len(primals_and_tangents) // 2
    primals, tangents = primals_and_tangents[:half], primals_and_tangents[half:]
    return _jvp_call_metafunc(primals, tangents, trace, detached=False, **kwargs)


jvp_impls[Transforms.IdentityOp] = _identity_call_jvp


def jvp(func):
    """Jacobian-vector product transform for a Thunder function.

    Args:
        func (Callable): A Thunder function to be transformed.

    Returns:
        Callable: A function that computes the Jacobian-vector product
            taking primals and tangents as arguments.
    """

    def wrapper(primals, tangents):
        trace = make_trace(func)(*primals)
        return jvp_call(primals, tangents, trace=trace)

    return wrapper


def jvp_eager(func, primals, tangents, executor="torch"):
    """Computes the Jacobian-vector product of a Thunder function.

    Args:
        func (Callable): A Thunder function to be transformed.
        primals (_type_): Primals of the function.
        tangents (_type_): Tangents of the function.
        executor (str, optional): Executor to use. Defaults to "torch".

    Returns:
        The result of the Jacobian-vector product.
    """
    trace = make_trace(func, executor=executor)(*primals)

    def jvp_func(*primals_and_tangents):
        _primals, _tangents = primals_and_tangents[: len(primals)], primals_and_tangents[len(primals) :]
        primals_tangents_pairs = safe_zip(_primals, _tangents)
        outs = eval_trace(trace, *primals_tangents_pairs, symbol_mapper=jvp_symbol_mapper)
        return outs

    jvp_trace = make_trace(jvp_func, executor=executor)(*primals, *tangents)
    jvp_traced = make_traced(partial(eval_trace, jvp_trace), executor=executor)
    return jvp_traced(*primals, *tangents)
