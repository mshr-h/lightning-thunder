from . import prims
from .. import make_trace
from ..executors.torch import ops_to_torch_ops_map
from .proxies import Proxy, TensorProxy
from .trace import Trace, get_trace, new_trace, reset_trace
from contextlib import contextmanager
from enum import auto, Enum
from functools import lru_cache
from typing import Any, Sequence


class Transforms(Enum):
    IdentityOp = auto()


def safe_map(f, *args):
    """Apply f to each element of args, which must all have the same length.

    Args:
        f: function to apply
        *args: arguments to apply f to

    Returns:
        list of results of applying f to each element of args
    """
    args = list(map(list, args))
    n = len(args[0])
    for arg in args[1:]:
        assert len(arg) == n, f"length mismatch: {list(map(len, args))}"
    return list(map(f, *args))


@lru_cache(maxsize=None)
def symbol_to_eval_map(symbol: prims.Prim):
    """Map a symbol to a function that evaluates it.

    Args:
        symbol: symbol to map
    """
    meta_func = prims.ops_to_meta_functions_map[symbol.op]

    prim_func = getattr(prims, symbol.name, None)
    if prim_func is not None:
        return prim_func

    def _fn(*args, **kwargs):
        t = get_trace()
        result = meta_func(*args, **kwargs)
        sym = prims.Prim(symbol.op, symbol.name, result, *args, **kwargs)
        t.add_symbol(sym)
        return result

    return _fn


def eval_trace(trace, *args, symbol_mapper=symbol_to_eval_map, **kwargs):
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
            return env[x] if type(x) is TensorProxy else x.value
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
        if not isinstance(symbol.result, Sequence):
            symbol.result = (symbol.result,)
        safe_map(write, symbol.result, result)

    if not isinstance(trace.outputs, Sequence):
        return read(trace.outputs)
    return safe_map(read, trace.outputs)


@contextmanager
def detached_trace():
    trace_token = new_trace()
    yield
    reset_trace(trace_token)


def _identity_call_metafunc(*args, trace: Trace):
    with detached_trace():
        return eval_trace(trace, *args)


identity_call = prims.make_prim(Transforms.IdentityOp, "identity_call", _identity_call_metafunc)


def identity(func):
    """Identity transform for a Thunder function.

    Args:
        func (Callable): A Thunder function to be transformed.
    """

    def wrapper(*args, **kwargs):
        trace = make_trace(func, executor="torch")(*args, **kwargs)
        return identity_call(*args, **kwargs, trace=trace)

    return wrapper


def _identity_call_pytorch(*args, trace: Trace):
    import torch

    def symbol_mapper(op):
        if op.op == Transforms.IdentityOp:
            return _identity_call_pytorch

        torch_op = ops_to_torch_ops_map[op.op]
        if isinstance(torch_op, str):
            return getattr(torch, torch_op.strip("pytorch."))
        return torch_op

    with detached_trace():
        return eval_trace(trace, *args, symbol_mapper=symbol_mapper)


ops_to_torch_ops_map[Transforms.IdentityOp] = _identity_call_pytorch
