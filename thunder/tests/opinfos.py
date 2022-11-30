import math
from collections import namedtuple
import pytest
from functools import partial

# TODO: make this import conditional on Torch being available and querying if should test
#   with torch
import torch
from torch.testing import make_tensor

import thunder.core.lang as tlang
import thunder.core.dtypes as datatypes
from thunder.langs.torch import torch_dtype

# Returns a noncontiguous (tensor with the same shape and values as t
# The noncontiguous tensor is constructed such that elements in the innermost
#   dimension are separated by zeros or (whenever possible) nans
# TODO: consider more complicated noncontiguity schemes
def noncontiguous_like(t):
    # Short-circuits if t is already noncontiguous
    if not t.is_contiguous():
        return t

    # Choose a "weird" value that won't be accessed
    if t.dtype.is_floating_point or t.dtype.is_complex:
        value = math.nan
    elif t.dtype == torch.bool:
        value = True
    else:
        value = 12

    result = t.new_empty(t.shape + (2,))
    result[..., 0] = value
    result[..., 1] = t.detach()
    result = result[..., 1]
    result.requires_grad_(t.requires_grad)
    return result


class SampleInput:
    """Represents sample inputs to a function."""

    __slots__ = [
        "args",
        "kwargs",
    ]

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    # Applies the transform f(t) -> t to each tensor and dtype in the SampleInput
    def transform(self, f):
        def tt(t):
            def _tt(t):
                with torch.no_grad():
                    return f(t)

            if isinstance(t, torch.Tensor):
                return _tt(t)
            elif isinstance(t, torch.dtype):
                return _tt(t)
            elif isinstance(t, list):
                return list(map(tt, t))
            elif isinstance(t, tuple):
                return tuple(map(tt, t))
            elif isinstance(t, dict):
                return {k: tt(v) for k, v in t.items()}
            else:
                return t

        return SampleInput(tt(self.args), tt(self.kwargs))

    def noncontiguous(self):
        def to_noncontiguous(t):
            if isinstance(t, torch.Tensor):
                return noncontiguous_like(t)
            elif isinstance(t, torch.dtype):
                return t

            return t

        return self.transform(to_noncontiguous)


# TODO: add executor
class DecorateInfo(object):
    """Describes which test, or type of tests, should be wrapped in the given
    decorator when testing an operator. Any test that matches all provided
    arguments will be decorated. The decorator will only be applied if the
    active_if argument is True."""

    __slots__ = [
        "decorator",
        "test_template_name",
        "devicetypes",
        "dtypes",
        "active_if",
    ]

    def __init__(
        self,
        decorator,
        test_template_name=None,
        *,
        devicetypes=None,
        dtypes=None,
        active_if=True,
    ):
        self.decorator = decorator
        self.test_template_name = test_template_name
        self.devicetypes = devicetypes
        self.dtypes = datatypes.resolve_dtypes(dtypes)
        self.active_if = active_if

    def is_active(self, test_template_name, devicetype, dtype):
        return (
            self.active_if
            and (self.test_template_name is None or self.test_template_name == test_template_name)
            and (self.devicetypes is None or devicetype in self.devicetypes)
            and (self.dtypes is None or dtype in self.dtypes)
        )


Domain = namedtuple("Domain", "low high")
opinfos = []

# TODO: require use of generic Thunder dtypes (once they exist)
class OpInfo:
    """Operator information and helper functions for acquiring it."""

    def __init__(
        self,
        op,
        *,
        name=None,
        device_types,
        dtypes,
        sample_input_generator,
        method_variant=None,
        operator_variant=None,
        torch_reference=None,
        numpy_reference=None,
        test_directives=(),
        domain=(None, None),
    ):
        self.op = op
        self.name = name if name is not None else op.__name__
        self._device_types = device_types
        self._dtypes = dtypes
        self.sample_input_generator = sample_input_generator
        self.method_variant = method_variant
        self.operator_variant = operator_variant
        self.torch_reference = torch_reference
        self.numpy_reference = numpy_reference
        self.test_directives = test_directives
        self.domain = Domain(*domain)

    def __call__(self, *args, **kwargs):
        """Calls the function variant of the operator."""
        return self.op(*args, **kwargs)

    # TODO: different sample inputs must be generated depending on the language context
    def sample_inputs(self, device_type, dtype, *, requires_grad=False, **kwargs):
        return self.sample_input_generator(self, device_type, dtype, requires_grad, **kwargs)

    def device_types(self):
        return set(self._device_types)

    def dtypes(self, device_type=None):
        if device_type is not None:
            raise NotImplementedError

        return datatypes.resolve_dtypes(self._dtypes)

    # TODO: add executor
    def test_decorators(self, test_name, devicetype, dtype):
        return [d.decorator for d in self.test_directives if d.is_active(test_name, devicetype, dtype)]


#
# Elementwise Unary OpInfos
#

# TODO: create elementwise unary OpInfo subclass and maybe auto add to list
elementwise_unary_ops = []


# TODO: add numbers
# TODO: add small value, large value, and extremal-valued samples
def elementwise_unary_generator(op, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=torch_dtype(dtype), low=op.domain.low, high=op.domain.high)

    shapes = (
        # TODO: restore size zero cases
        # (0, 2, 1),
        # (5, 0, 3),
        (),
        (11,),
        (4, 4),
        (1024, 1024),
        (64, 64, 64),
    )

    # Typical inputs
    for shape in shapes:
        yield SampleInput(make_arg(shape))

    # Noncontiguous inputs
    for shape in shapes:
        yield SampleInput(make_arg(shape, noncontiguous=True))

    # Arbitrarily strided inputs
    # shape, strides, offset
    strided_cases = (
        ((5, 6, 2), (1, 1, 7), 2),
        ((5, 5, 4), (1, 1, 7), 2),
        ((5, 5, 2), (4, 5, 7), 3),
        ((5, 5, 2), (5, 5, 7), 3),
        ((5, 5, 2), (5, 5, 5), 3),
        ((9, 5, 2), (0, 1, 7), 3),
    )

    for shape, strides, offset in strided_cases:
        a = make_arg(
            500,
        ).as_strided(shape, strides, offset)
        yield SampleInput(a)


abs_opinfo = OpInfo(
    tlang.abs,
    device_types=("cpu", "cuda"),
    # TODO check types we support
    dtypes=(datatypes.exact, datatypes.inexact),
    sample_input_generator=elementwise_unary_generator,
    torch_reference=torch.abs,
)
elementwise_unary_ops.append(abs_opinfo)

acos_opinfo = OpInfo(
    tlang.acos,
    domain=(-1, 1),
    device_types=("cpu", "cuda"),
    # TODO check types we support
    dtypes=(datatypes.exact, datatypes.inexact),
    sample_input_generator=elementwise_unary_generator,
    torch_reference=torch.acos,
)
elementwise_unary_ops.append(acos_opinfo)


# Puts all opinfos into the "opinfos" list
opinfos.extend(elementwise_unary_ops)


#
# Elementwise Binary OpInfos
#

# TODO: create elementwise binary OpInfo subclass and maybe auto add to list
elementwise_binary_ops = []


# TODO: extend this generator
def elementwise_binary_generator(op, device, dtype, requires_grad, **kwargs):
    a = make_tensor((4, 4), device=device, dtype=torch_dtype(dtype))
    b = make_tensor((4, 4), device=device, dtype=torch_dtype(dtype))

    yield SampleInput(a, b)

    # Tests broadcasting
    c = make_tensor((4, 1), device=device, dtype=torch_dtype(dtype))
    yield SampleInput(a, c)


# TODO: update dtypes with Thunder dtypes (when they exist)
add_opinfo = OpInfo(
    tlang.add,
    device_types=("cpu", "cuda"),
    dtypes=(datatypes.exact, datatypes.inexact),
    sample_input_generator=elementwise_binary_generator,
    torch_reference=torch.add,
)
elementwise_binary_ops.append(add_opinfo)

# NOTE: nvFuser does not currently support uint8, int8, or int16
bitwise_and_opinfo = OpInfo(
    tlang.bitwise_and,
    device_types=("cpu", "cuda"),
    dtypes=(datatypes.exact,),
    sample_input_generator=elementwise_binary_generator,
    torch_reference=torch.bitwise_and,
)
elementwise_binary_ops.append(bitwise_and_opinfo)

mul_opinfo = OpInfo(
    tlang.mul,
    device_types=("cpu", "cuda"),
    dtypes=(datatypes.exact, datatypes.inexact),
    sample_input_generator=elementwise_binary_generator,
    torch_reference=torch.mul,
)
elementwise_binary_ops.append(mul_opinfo)

sub_opinfo = OpInfo(
    tlang.sub,
    device_types=("cpu", "cuda"),
    dtypes=(datatypes.exact, datatypes.inexact),
    sample_input_generator=elementwise_binary_generator,
    torch_reference=torch.sub,
    test_directives=(DecorateInfo(pytest.mark.xfail, "test_core_vs_torch_consistency", dtypes=(datatypes.bool8,)),),
)
elementwise_binary_ops.append(sub_opinfo)


# Puts all opinfos into the "opinfos" list
opinfos.extend(elementwise_binary_ops)
