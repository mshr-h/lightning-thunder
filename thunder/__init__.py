import os
from collections import deque
from functools import wraps
from typing import Callable, Sequence, Optional
import time

import thunder.langs as langs
import thunder.core.dtypes as dtypes
from thunder.__about__ import *

from .core.trace import (
    get_trace,
    new_trace,
    reset_executor_context,
    reset_language_context,
    reset_trace,
    set_executor_context,
    get_executor_context,
    set_language_context,
)

_PACKAGE_ROOT = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.dirname(_PACKAGE_ROOT)

__all__ = [
    # dtype aliases
    "bool8",
    "uint8",
    "int8",
    "int16",
    "int32",
    "int64",
    "bfloat16",
    "float16",
    "float32",
    "float64",
    "complex32",
    "complex64",
    "complex128",
    # tracing functions
    "make_traced",
]

#
# dtype aliases
#
bool8 = dtypes.bool8
uint8 = dtypes.uint8
int8 = dtypes.int8
int16 = dtypes.int16
int32 = dtypes.int32
int64 = dtypes.int64
bfloat16 = dtypes.bfloat16
float16 = dtypes.float16
float32 = dtypes.float32
float64 = dtypes.float64
complex32 = dtypes.complex32
complex64 = dtypes.complex64
complex128 = dtypes.complex128

#
# tracing functions
#


def _get_executor(executor=None):
    if executor is None:
        ex = get_executor_context()
        if ex is None:
            raise ValueError("No executor specified!")
        return ex

    if executor == "torch":
        try:
            from .executors.torch import torchCtx

            return torchCtx()
        except ModuleNotFoundError:
            raise RuntimeError(
                "The 'torch' executor was requested, but the `torch` package "
                "is not available. Please make sure the `torch` package is installed"
                "in the environment."
            )

    if executor == "nvfuser":
        try:
            from .executors.nvfuser import nvFuserCtx

            return nvFuserCtx()
        except ModuleNotFoundError:
            raise RuntimeError(
                "The 'nvfuser' executor was requested, but NVFuser is not available. "
                "Please make sure the `torch` package is installed and CUDA is available."
            )

    raise ValueError(f"Trying to acquire an executor from unknown object {executor}!")


def _construct_trace(*args, fn, lang_ctx, **kwargs):
    t = get_trace()

    # Constructs proxies
    proxy_args = deque()
    for arg in args:
        # NOTE: a very limited exception to the requirement we can proxy all inputs
        # TODO: consider more carefully what we proxy vs. don't and how it's modeled
        if isinstance(arg, Sequence):
            proxy_args.append(arg)
            continue

        p = lang_ctx.proxy(arg)
        t.add_input(p)
        proxy_args.append(p)

    proxy_kwargs = {}
    for k, v in kwargs.items():
        # NOTE: two ugly exceptions to what we proxy -- kwarg strings and dtypes are passed through
        # TODO: consider more carefully what we proxy vs. don't
        if isinstance(v, str):
            proxy_kwargs[k] = v
        elif lang_ctx.is_dtype(v):
            proxy_kwargs[k] = lang_ctx.thunder_dtype(v)
        else:
            p = lang_ctx.proxy(v)
            t.add_kwarg_input(k, p)
            proxy_kwargs[k] = p

    # TODO: support multiple return values
    proxy_result = fn(*proxy_args, **proxy_kwargs)
    t.add_output(proxy_result)

    return t


def make_traced(fn: Callable, executor: Optional[str] = None, language_ctx=langs.torch, _info=False) -> Callable:
    """Converts a callable in a callable that will be traced and then executed.

    Example usage:

      def foo(a, b):
        return tlang.add(a, b)

      traced_foo = thunder.make_traced(foo)

      a = torch.randn(2, 2, device='cuda')
      b = torch.randn(2, 1, device='cuda')

      result = traced_foo(a, b)
    """

    ex = _get_executor(executor)
    lang_ctx = language_ctx.ctx()

    @wraps(fn)
    def _fn(*args, **kwargs):
        acquisition_start = time.time_ns()

        # Sets the proper tracing context
        trace_token = new_trace()
        executor_token = set_executor_context(ex)
        lang_token = set_language_context(lang_ctx)

        proxyargs, proxykwargs = _make_proxies(fn, langctx, *args, **kwargs)

        trace = _construct_trace(*args, fn=fn, lang_ctx=lang_ctx, **kwargs)

        # Resets the tracing context
        reset_trace(trace_token)
        reset_language_context(lang_token)
        if executor_token is not None:
            reset_executor_context(executor_token)

        acquisition_end = time.time_ns()

        # print(trace)

        result, meta = ex.execute(trace, *args, **kwargs)

        # TODO: convert nvFuser output to appropriate object based on language ctx
        # TODO: if the output is a datastructure it will be flattened before being handed to the executor
        #   this needs to re-wrap the executor outputs into the datstructure
        if len(result) == 1:
            # Hack to unwrap singleton results
            result = result[0]

        if _info:
            meta.update(
                {
                    "acquisition_time": acquisition_end - acquisition_start,
                }
            )
            return result, meta

        return result

    return _fn
