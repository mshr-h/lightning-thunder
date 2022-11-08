from typing import Callable
from functools import wraps
from collections import deque

from .core import lang
from .core import trace
from .core.trace import get_trace, new_trace, reset_trace
from .core.proxies import proxy
from .executors.nvfuser import execute as nvfuser

# TODO make this dependency optional
import torch

__all__ = [
    "make_traced",
]

# TODO: currently assumes all inputs are proxyable and
#     a single (proxy) tensor is returned from the operation
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

        # TODO: add support for kwargs
        if len(kwargs) > 0:
            raise NotImplementedError

        # Acquires a new tracing context
        trace_token = new_trace()
        t = get_trace()

        # Constructs proxies
        proxies = deque()
        for arg in args:

            # TODO: consider handling types that we won't proxy
            p = proxy(arg)
            inp = t.add_input(p)
            proxies.append(p)

        # TODO: support multiple return values
        proxy_result = fn(*proxies)
        t.add_output(proxy_result)

        nv_result, fusion = nvfuser(t, *args)

        reset_trace(trace_token)
        return nv_result

    return _fn
