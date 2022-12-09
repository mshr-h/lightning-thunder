import inspect
import os
import sys

from itertools import product
from functools import wraps

import thunder.core.dtypes as dtypes
from thunder.core.trace import set_executor_context, reset_executor_context

__all__ = [
    "ops",
]


# TODO: Add device type functionality to an object in this list
def _all_device_types():
    return ("cpu", "cuda")


def _available_device_types():
    try:
        import torch

        if torch.cuda.is_available():
            return ("cpu", "cuda")
        return ("cpu",)
    except ModuleNotFoundError:
        ("cpu",)


class Executor(object):
    def supports_datatype(self, dtype):
        return dtype in dtypes.resolve_dtypes(self.supported_datatypes)

    def supports_devicetype(self, devicetype):
        return devicetype in self.supported_devicetypes


class nvFuser(Executor):
    name = "nvFuser"
    supported_devicetypes = ("cuda",)
    supported_datatypes = (
        dtypes.floating,
        dtypes.bool8,
        dtypes.int32,
        dtypes.int64,
        dtypes.complex64,
        dtypes.complex128,
    )

    ctx = None

    def get_executor_context(self):
        if self.ctx is None:
            from thunder.executors.nvfuser import nvFuserCtx

            self.ctx = nvFuserCtx()

        return self.ctx


class TorchEx(Executor):
    name = "TorchEx"
    supported_devicetypes = ("cpu", "cuda")
    supported_datatypes = (dtypes.datatype,)

    ctx = None

    def get_executor_context(self):
        if self.ctx is None:
            from thunder.executors.torch import torchCtx

            self.ctx = torchCtx()

        return self.ctx


def _all_executors():
    executors = []

    try:
        import torch

        executors.append(TorchEx())
    except ModuleNotFoundError:
        pass

    try:
        import torch._C._nvfuser

        executors.append(nvFuser())
    except ModuleNotFoundError:
        pass

    return executors


# TODO: add decorator support, support for test directives -- how would this control assert_close behavior?
def _instantiate_test_template(template, scope, *, opinfo, executor, device, dtype):
    """Instanties a test template for an operator."""

    # Ex. test_foo_CUDA_float32
    test_name = "_".join((template.__name__, opinfo.name, executor.name, device.upper(), str(dtype)))

    def test():
        # TODO: currently this passes the device type as a string, but actually a device or multiple devices
        #   should be passed to the test
        tok = set_executor_context(executor.get_executor_context())
        result = template(opinfo, device, dtype)
        reset_executor_context(tok)
        return result

    # TODO: pass device type explicitly
    for decorator in opinfo.test_decorators(template.__name__, executor, device, dtype):
        test = decorator(test)

    # Mimics the instantiated test
    # TODO: review this mimicry -- are there other attributes to mimic?
    test.__name__ = test_name
    test.__module__ = test.__module__

    return test


# TODO: don't pass the device type to the test, select an actual device
class ops:

    # TODO: support other kinds of dtype specifications
    def __init__(
        self, opinfos, *, supported_executors=None, supported_device_types=None, supported_dtypes=None, scope=None
    ):
        self.opinfos = opinfos

        self.supported_executors = (
            set(supported_executors) if supported_executors is not None else set(_all_executors())
        )

        self.supported_device_types = (
            set(supported_device_types) if supported_device_types is not None else set(_all_device_types())
        )
        self.supported_dtypes = (
            dtypes.resolve_dtypes(supported_dtypes) if supported_dtypes is not None else dtypes.all_datatypes
        )

        # Acquires the caller's global scope
        if scope is None:
            previous_frame = inspect.currentframe().f_back
            scope = previous_frame.f_globals
        self.scope = scope

    def __call__(self, test_template):
        # NOTE: unlike a typical decorator, this __call__ does not return a function, because it may
        #   (and typically does) instantiate multiple functions from the template it consumes
        #   Since Python doesn't natively support one-to-many function decorators, the produced
        #   functions are directly assigned to the requested scope (the caller's global scope by default)

        for opinfo in self.opinfos:
            device_types = (
                opinfo.device_types()
                .intersection(self.supported_device_types)
                .intersection(set(_available_device_types()))
            )
            for executor, devicetype in product(self.supported_executors, device_types):
                if not executor.supports_devicetype(devicetype):
                    continue

                # TODO: pass device_type to dtypes()
                dtypes = opinfo.dtypes()
                if self.supported_dtypes is not None:
                    dtypes = dtypes.intersection(self.supported_dtypes)

                for dtype in dtypes:
                    if not executor.supports_datatype(dtype):
                        continue

                    test = _instantiate_test_template(
                        test_template,
                        self.scope,
                        opinfo=opinfo,
                        executor=executor,
                        device=devicetype,
                        dtype=dtype,
                    )
                    # Adds the instantiated test to the requested scope
                    self.scope[test.__name__] = test


def run_snippet(snippet, opinfo, device_type, dtype, *args, **kwargs):
    try:
        snippet(*args, **kwargs)
    except Exception as e:
        exc_info = sys.exc_info()

        # Raises exceptions that occur with pytest, and returns debug information when
        # called otherwise
        # NOTE: PYTEST_CURRENT_TEST is set by pytest
        if "PYTEST_CURRENT_TEST" in os.environ:
            raise e
        return e, exc_info, snippet, opinfo, device_type, dtype, args, kwargs

    return None
