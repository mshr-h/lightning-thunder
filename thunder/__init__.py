from typing import Callable, Sequence
from functools import wraps
from collections import deque

from .core import lang
from .core import trace
from .core.trace import (
    get_trace,
    new_trace,
    reset_trace,
    set_language_context,
    get_language_context,
    reset_language_context,
    set_executor_context,
    get_executor_context,
    reset_executor_context,
)
from .core.proxies import proxy
from .executors.nvfuser import execute as nvfuser, nvFuserCtx

# TODO make this dependency optional
import torch

__all__ = [
    "make_traced",
]


def make_traced(fn: Callable) -> Callable:
    """
    Converts a callable in a callable that will be traced and then executed.

    Example usage:

      def foo(a, b):
        return tlang.add(a, b)

      traced_foo = thunder.make_traced(foo)

      a = torch.randn(2, 2, device='cuda')
      b = torch.randn(2, 1, device='cuda')

      result = traced_foo(a, b)

    """

    @wraps(fn)
    def _fn(*args, **kwargs):
        # Acquires a new tracing context
        trace_token = new_trace()
        executor_token = set_executor_context(nvFuserCtx())
        lang_token = set_language_context(lang.CoreLangCtx())
        t = get_trace()

        # Constructs proxies
        proxy_args = deque()
        for arg in args:
            # NOTE: a very limited exception to the requirement we can proxy all inputs
            # TODO: consider more carefully what we proxy vs. don't and how it's modeled
            if isinstance(arg, Sequence):
                proxy_args.append(arg)
                continue

            p = proxy(arg)
            inp = t.add_input(p)
            proxy_args.append(p)

        proxy_kwargs = {}
        for k, v in kwargs.items():
            p = proxy(v)
            inp = t.add_kwarg_input(k, p)
            proxy_kwargs[k] = p

        # TODO: support multiple return values
        proxy_result = fn(*proxy_args, **proxy_kwargs)
        t.add_output(proxy_result)

        print(t)

        nv_result, fusion = nvfuser(t, *args, **kwargs)

        reset_trace(trace_token)
        reset_language_context(lang_token)
        reset_executor_context(executor_token)

        # TODO: if the output is a datastructure it will be flattened before being handed to nvFuser
        #   this needs to re-wrap the nvFuser outputs into the datstructure
        if len(nv_result) == 1:
            # Hack to unwrap singleton results
            return nv_result[0]

        return nv_result

    return _fn
