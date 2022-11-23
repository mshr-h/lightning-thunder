import os
from itertools import product
import inspect
import sys


__all__ = [
    "ops",
]


# TODO: Add device type functionality to an object in this list
_all_device_types = ["cpu", "cuda"]

# _device_types_executor_support_map = {
#     "torch": ("cpu", "cuda"),
#     "nvfuser": ("cuda", )
# }

# TODO: add decorator support, support for test directives -- how would this control assert_close behavior?
def _instantiate_test_template(template, scope, *, opinfo, device, dtype):
    """
    Instanties a test template for an operator.
    """

    # Ex. test_foo_CUDA_float32
    test_name = "_".join(
        (template.__name__, opinfo.name, device.upper(), str(dtype))
    )

    def test():
        # TODO: currently this passes the device type as a string, but actually a device or multiple devices
        #   should be passed to the test
        result = template(opinfo, device, dtype)
        return result

    # Mimics the instantiated test
    # TODO: review this mimicry -- are there other attributes to mimic?
    test.__name__ = test_name
    test.__module__ = test.__module__

    return test


# TODO: don't pass the device type to the test, select an actual device
class ops(object):

    # TODO: support other kinds of dtype specifications
    def __init__(
        self, opinfos, *, supported_device_types=None, supported_dtypes=None, scope=None
    ):
        self.opinfos = opinfos
        self.supported_device_types = (
            set(supported_device_types)
            if supported_device_types is not None
            else set(_all_device_types)
        )
        self.supported_dtypes = (
            set(supported_dtypes) if supported_dtypes is not None else None
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
            device_types = opinfo.device_types().intersection(
                self.supported_device_types
            )
            for device_type in device_types:
                # TODO: pass device_type to dtypes()

                dtypes = opinfo.dtypes()
                if self.supported_dtypes is not None:
                    dtypes = dtypes.intersection(self.supported_dtypes)

                for dtype in dtypes:
                    test = _instantiate_test_template(
                        test_template,
                        self.scope,
                        opinfo=opinfo,
                        device=device_type,
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
