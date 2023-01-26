from torch.testing import assert_close, make_tensor

import thunder
import thunder.core.lang as tlang
import thunder.langs.torch as ttorch

from .framework import executors, ops, run_snippet
from .opinfos import elementwise_binary_ops, elementwise_unary_ops

# Tests for elementwise binary operators

# TODO: test that the operator variant works properly
# TODO: generate the following tests using opinfos (more number sample inputs needed)


@executors(dtypes=(thunder.float32,))
def test_abs_integer(executor, device, dtype):
    def foo(a, b):
        a_abs = tlang.abs(a)
        return tlang.add(a_abs, b)

    traced_foo = thunder.make_traced(foo, executor=executor)

    tdtype = ttorch.torch_dtype(dtype)
    a = -3
    b = make_tensor((1, 8), device=device, dtype=tdtype)

    thunder_result = traced_foo(a, b)
    torch_result = 3 + b
    assert_close(thunder_result, torch_result)


@executors(dtypes=(thunder.float32,))
def test_abs_float(executor, device, dtype):
    def foo(a, b):
        a_abs = tlang.abs(a)
        return tlang.add(a_abs, b)

    traced_foo = thunder.make_traced(foo, executor=executor)

    tdtype = ttorch.torch_dtype(dtype)
    a = -2.7
    b = make_tensor((1, 8), device=device, dtype=tdtype)

    thunder_result = traced_foo(a, b)
    torch_result = abs(a) + b
    assert_close(thunder_result, torch_result)


@executors(dtypes=(thunder.float32,))
def test_core_tensor_methods(executor, device, dtype):
    def foo(a, b, c, d):
        return a + b - c + (d - a)

    traced_foo = thunder.make_traced(foo, executor=executor)

    tdtype = ttorch.torch_dtype(dtype)
    a = make_tensor((4, 4), device=device, dtype=tdtype)
    b = make_tensor((2, 1, 4), device=device, dtype=tdtype)
    c = make_tensor((4, 1), device=device, dtype=tdtype)
    d = make_tensor((1, 1, 4), device=device, dtype=tdtype)

    thunder_result = traced_foo(a, b, c, d)
    torch_result = a + b - c + (d - a)
    assert_close(thunder_result, torch_result)


@executors(dtypes=(thunder.float32,))
def test_add_integer_constant(executor, device, dtype):
    def foo(a, b):
        c = tlang.add(a, 2)
        return tlang.add(c, b)

    traced_foo = thunder.make_traced(foo, executor=executor)

    tdtype = ttorch.torch_dtype(dtype)
    a = make_tensor((2, 1), device=device, dtype=tdtype)
    b = make_tensor((1, 2), device=device, dtype=tdtype)

    thunder_result = traced_foo(a, b)
    torch_result = (a + 2) + b

    assert_close(thunder_result, torch_result)


@executors(dtypes=(thunder.float32,))
def test_add_integer_input(executor, device, dtype):
    def foo(a, b):
        return tlang.add(a, b)

    traced_foo = thunder.make_traced(foo, executor=executor)

    tdtype = ttorch.torch_dtype(dtype)
    a = make_tensor((2, 1), device=device, dtype=tdtype)

    thunder_result = traced_foo(a, 3)
    torch_result = a + 3

    assert_close(thunder_result, torch_result)


@executors(dtypes=(thunder.float32,))
def test_add_integer_inputs(executor, device, dtype):
    def foo(a, b, c):
        d = tlang.add(a, b)
        return tlang.add(c, d)

    traced_foo = thunder.make_traced(foo, executor=executor)

    tdtype = ttorch.torch_dtype(dtype)
    a = make_tensor((3, 2), device=device, dtype=tdtype)

    thunder_result = traced_foo(3, 4, a)
    torch_result = 3 + 4 + a
    assert_close(thunder_result, torch_result)


@executors(dtypes=(thunder.float32,))
def test_add_integer_constants(executor, device, dtype):
    def foo(a):
        b = tlang.add(2, 3)
        return tlang.add(a, b)

    traced_foo = thunder.make_traced(foo, executor=executor)

    tdtype = ttorch.torch_dtype(dtype)
    a = make_tensor((2, 4), device=device, dtype=tdtype)

    thunder_result = traced_foo(a)
    torch_result = 5 + a
    assert_close(thunder_result, torch_result)


@executors(dtypes=(thunder.float32,))
def test_add_floats(executor, device, dtype):
    def foo(a, b):
        c = tlang.add(2.0, a)
        return tlang.add(b, c)

    traced_foo = thunder.make_traced(foo, executor=executor)

    tdtype = ttorch.torch_dtype(dtype)
    a = make_tensor((2, 4), device=device, dtype=tdtype)

    thunder_result = traced_foo(0.7, a)
    torch_result = 2.0 + 0.7 + a
    assert_close(thunder_result, torch_result)
