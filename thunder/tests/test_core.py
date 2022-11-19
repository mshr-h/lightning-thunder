from itertools import product
import pytest

import torch
import thunder
import thunder.core.lang as tlang
import thunder.core.utils as utils

import thunder.langs.torch as ttorch

from torch.testing import make_tensor, assert_close

# TODO: sample across executor_types and devices
from thunder.tests import executor_type, device


# TODO: add device/dtype instantiation
# TODO: use OpInfo samples
def test_add():
    def foo(a, b):
        return tlang.add(a, b)

    traced_foo = thunder.make_traced(foo, executor=executor_type)

    a = torch.testing.make_tensor((2, 2), device=device, dtype=torch.float32)
    b = torch.testing.make_tensor((2, 2), device=device, dtype=torch.float32)

    thunder_result = traced_foo(a, b)
    torch_result = a + b

    torch.testing.assert_close(thunder_result, torch_result)


def test_add_broadcast():
    def foo(a, b):
        return tlang.add(a, b)

    traced_foo = thunder.make_traced(foo, executor=executor_type)

    a = torch.testing.make_tensor((2, 1), device=device, dtype=torch.float32)
    b = torch.testing.make_tensor((1, 2), device=device, dtype=torch.float32)

    thunder_result = traced_foo(a, b)
    torch_result = a + b

    torch.testing.assert_close(thunder_result, torch_result)


def test_add_integer_constant():
    def foo(a, b):
        c = tlang.add(a, 2)
        return tlang.add(c, b)

    traced_foo = thunder.make_traced(foo, executor=executor_type)

    a = torch.testing.make_tensor((2, 1), device=device, dtype=torch.float32)
    b = torch.testing.make_tensor((1, 2), device=device, dtype=torch.float32)

    thunder_result = traced_foo(a, b)
    torch_result = (a + 2) + b

    torch.testing.assert_close(thunder_result, torch_result)


def test_add_integer_input():
    def foo(a, b):
        return tlang.add(a, b)

    traced_foo = thunder.make_traced(foo, executor=executor_type)

    a = make_tensor((2, 1), device=device, dtype=torch.float32)

    thunder_result = traced_foo(a, 3)
    torch_result = a + 3

    assert_close(thunder_result, torch_result)


def test_add_integer_inputs():
    def foo(a, b, c):
        d = tlang.add(a, b)
        return tlang.add(c, d)

    traced_foo = thunder.make_traced(foo, executor=executor_type)

    a = make_tensor((3, 2), device=device, dtype=torch.float32)

    thunder_result = traced_foo(3, 4, a)
    torch_result = 3 + 4 + a
    assert_close(thunder_result, torch_result)


def test_add_integer_constants():
    def foo(a):
        b = tlang.add(2, 3)
        return tlang.add(a, b)

    traced_foo = thunder.make_traced(foo, executor=executor_type)

    a = make_tensor((2, 4), device=device, dtype=torch.float32)

    thunder_result = traced_foo(a)
    torch_result = 5 + a
    assert_close(thunder_result, torch_result)


def test_add_floats():
    def foo(a, b):
        c = tlang.add(2.0, a)
        return tlang.add(b, c)

    traced_foo = thunder.make_traced(foo, executor=executor_type)

    a = make_tensor((2, 4), device=device, dtype=torch.float32)

    thunder_result = traced_foo(0.7, a)
    torch_result = 2.0 + 0.7 + a
    assert_close(thunder_result, torch_result)


def test_sub():
    def foo(a, b):
        return tlang.sub(a, b)

    traced_foo = thunder.make_traced(foo, executor=executor_type)

    a = torch.testing.make_tensor((2, 2), device=device, dtype=torch.float32)
    b = torch.testing.make_tensor((2, 2), device=device, dtype=torch.float32)

    thunder_result = traced_foo(a, b)
    torch_result = a - b

    torch.testing.assert_close(thunder_result, torch_result)


def test_true_divide():
    def foo(a, b):
        return tlang.true_divide(a, b)

    traced_foo = thunder.make_traced(foo, executor=executor_type)

    a = torch.testing.make_tensor((2, 2), device=device, dtype=torch.float32)
    b = torch.testing.make_tensor((2, 2), device=device, dtype=torch.float32)

    thunder_result = traced_foo(a, b)
    torch_result = a / b

    torch.testing.assert_close(thunder_result, torch_result)


def test_integer_isinstance_mimicry():
    # isinstance() works as expected
    def foo(a, b, c):
        if isinstance(a, int):
            return tlang.add(a, b)

        return tlang.add(b, c)

    traced_foo = thunder.make_traced(foo, executor=executor_type)

    a = make_tensor((2, 1), device=device, dtype=torch.float32)
    b = make_tensor((2, 2), device=device, dtype=torch.float32)
    c = make_tensor((1, 2), device=device, dtype=torch.float32)

    thunder_result = traced_foo(a, b, c)
    torch_result = b + c
    assert_close(thunder_result, torch_result)

    thunder_result = traced_foo(2, b, c)
    torch_result = 2 + b
    assert_close(thunder_result, torch_result)

    # type() doesn't work (it returns the actual type)
    def bar(a, b, c):
        if type(a) is int:
            return tlang.add(a, b)

        return tlang.add(b, c)

    traced_bar = thunder.make_traced(bar, executor=executor_type)

    try:
        thunder_result = traced_bar(a, b, c)
        torch_result = b + c
        assert_close(thunder_result, torch_result)
        pytest.fail()
    except:
        pass

    try:
        thunder_result = traced_bar(2, b, c)
        torch_result = 2 + b
        assert_close(thunder_result, torch_result)
        pytest.fail()
    except:
        pass


# FIXME NVIDIA: this test will cause a segmentation fault!
# def test_return_integer():
#     def foo(a, b):
#         return tlang.add(a, b)

#     traced_foo = thunder.make_traced(foo, executor=executor_type)

#     thunder_result = traced_foo(3, 4)
#     python_result = 3 + 4
#     assert_close(thunder_result, python_result)


def test_abs():
    def foo(a):
        return tlang.abs(a)

    traced_foo = thunder.make_traced(foo, executor=executor_type)

    a = make_tensor((2, 8), device=device, dtype=torch.float32)

    thunder_result = traced_foo(a)
    torch_result = torch.abs(a)
    assert_close(thunder_result, torch_result)


def test_abs_integer():
    def foo(a, b):
        a_abs = tlang.abs(a)
        return tlang.add(a_abs, b)

    traced_foo = thunder.make_traced(foo, executor=executor_type)

    a = -3
    b = make_tensor((1, 8), device=device, dtype=torch.float32)

    thunder_result = traced_foo(a, b)
    torch_result = 3 + b
    assert_close(thunder_result, torch_result)


def test_abs_float():
    def foo(a, b):
        a_abs = tlang.abs(a)
        return tlang.add(a_abs, b)

    traced_foo = thunder.make_traced(foo, executor=executor_type)

    a = -2.7
    b = make_tensor((1, 8), device=device, dtype=torch.float32)

    thunder_result = traced_foo(a, b)
    torch_result = abs(a) + b
    assert_close(thunder_result, torch_result)


def test_elementwise_binary_prim_shape_mismatch():
    pass


def test_elementwise_binary_prim_dtype_mismatch():
    pass


def test_torch_var():
    # Tests passing all arguments as function inputs
    def foo(a, dim, *, keepdim=False, correction=1):
        return ttorch.var(a, dim, keepdim=keepdim, correction=correction)

    traced_foo = thunder.make_traced(foo, executor=executor_type)

    a = torch.testing.make_tensor((4, 4), device=device, dtype=torch.float32)

    # Full reduction
    thunder_result = traced_foo(a, [0, 1])
    torch_result = torch.var(a, [0, 1])
    assert_close(thunder_result, torch_result)

    # Reduce along dim 1
    thunder_result = traced_foo(a, [1])
    torch_result = torch.var(a, [1])
    assert_close(thunder_result, torch_result)

    # TODO: review with NVIDIA -- how should correction be passed?
    # Specifying the correction
    # thunder_result = traced_foo(a, [1], correction=2)
    # torch_result = torch.var(a, [1], correction=2)
    # assert_close(thunder_result, torch_result)

    # # Specifying keepdim
    # thunder_result = traced_foo(a, [1], keepdim=True, correction=2)
    # torch_result = torch.var(a, [1], keepdim=True, correction=2)
    # assert_close(thunder_result, torch_result)

    # Tests passing arguments as constants
    def foo(a):
        return ttorch.var(a, [0, 1], keepdim=True, correction=2)

    traced_foo = thunder.make_traced(foo, executor=executor_type)

    a = torch.testing.make_tensor((4, 4), device=device, dtype=torch.float32)

    thunder_result = traced_foo(a)
    torch_result = torch.var(a, [0, 1], keepdim=True, correction=2)
    assert_close(thunder_result, torch_result)


def test_torch_mean():
    def foo(a, dim=None, keepdim=False, *, dtype=None):
        return ttorch.mean(a, dim, keepdim, dtype=dtype)

    traced_foo = thunder.make_traced(foo, executor=executor_type)

    a = torch.testing.make_tensor((4, 4), device=device, dtype=torch.float32)

    # Full reduction
    thunder_result = traced_foo(a, [0, 1])
    torch_result = torch.mean(a, [0, 1])
    assert_close(thunder_result, torch_result)

    # Reduce along dim 1
    thunder_result = traced_foo(a, [1])
    torch_result = torch.mean(a, [1])
    assert_close(thunder_result, torch_result)


def test_var_mean():
    def foo(a, dim=None, unbiased=None, keepdim=False, *, correction=None):
        return ttorch.var_mean(a, dim, unbiased, keepdim=keepdim, correction=correction)

    traced_foo = thunder.make_traced(foo, executor=executor_type)

    a = torch.testing.make_tensor((4, 4), device=device, dtype=torch.float32)

    # Full reduction
    thunder_result = traced_foo(a, [0, 1])
    torch_result = torch.var_mean(a, [0, 1])
    assert_close(thunder_result, torch_result)

    # Reduce along dim 1
    thunder_result = traced_foo(a, [1])
    torch_result = torch.var_mean(a, [1])
    assert_close(thunder_result, torch_result)

    # Tests passing arguments as constants
    def foo(a):
        return ttorch.var_mean(a, [0, 1], keepdim=True, correction=2)

    traced_foo = thunder.make_traced(foo, executor=executor_type)

    a = torch.testing.make_tensor((4, 4), device=device, dtype=torch.float32)

    thunder_result = traced_foo(a)
    torch_result = torch.var_mean(a, [0, 1], keepdim=True, correction=2)
    assert_close(thunder_result, torch_result)


def test_core_tensor_methods():
    def foo(a, b, c, d):
        return a + b - c + (d - a)

    traced_foo = thunder.make_traced(foo, executor=executor_type)

    a = torch.testing.make_tensor((4, 4), device=device, dtype=torch.float32)
    b = torch.testing.make_tensor((2, 1, 4), device=device, dtype=torch.float32)
    c = torch.testing.make_tensor((4, 1), device=device, dtype=torch.float32)
    d = torch.testing.make_tensor((1, 1, 4), device=device, dtype=torch.float32)

    thunder_result = traced_foo(a, b, c, d)
    torch_result = a + b - c + (d - a)
    assert_close(thunder_result, torch_result)


# TODO: this test just spot-checks type promotion -- it could probably be better
def test_type_promotion():
    def foo(a, b):
        return a + b

    traced_foo = thunder.make_traced(foo, executor=executor_type)

    b1 = make_tensor((2, 2), device=device, dtype=torch.bool)
    i64 = make_tensor((2, 2), device=device, dtype=torch.int64)
    bf16 = make_tensor((2, 2), device=device, dtype=torch.bfloat16)
    f16 = make_tensor((2, 2), device=device, dtype=torch.float16)
    f32 = make_tensor((2, 2), device=device, dtype=torch.float32)

    # float16 x float16 type promotion -- float16 result dtype
    result = traced_foo(f16, f16)
    assert result.dtype is torch.float16

    # float16 x float32 type promotion -- float32 result dtype
    result = traced_foo(f16, f32)
    assert result.dtype is torch.float32

    # float16 x bfloat16 type promotion -- float32 result dtype
    result = traced_foo(f16, bf16)
    assert result.dtype is torch.float32

    # int64 x float16 type promotion -- float16 result dtype
    result = traced_foo(f16, i64)
    assert result.dtype is torch.float16

    # bool x int64 type promotion -- int64 result dtype
    result = traced_foo(b1, i64)
    assert result.dtype is torch.int64

    # f x int64 type promotion -- float result dtype
    result = traced_foo(2.0, i64)
    assert result.dtype is torch.float32

    # b1 x int64 type promotion -- int64 result dypte
    result = traced_foo(b1, i64)
    assert result.dtype is torch.int64

    def bar(a, b, c):
        return a - b + c

    traced_bar = thunder.make_traced(bar, executor=executor_type)

    # float x int64 x float16 type promotion -- float16 result dtype
    result = traced_bar(2.0, i64, f16)
    assert result.dtype is torch.float16

    # float x int x int64 -- float32 result dtype
    result = traced_bar(2.1, -1, i64)
    assert result.dtype is torch.float32


def test_int_to_float_type_promotion():
    def foo(a, b):
        return a / b

    traced_foo = thunder.make_traced(foo, executor=executor_type)

    i64 = make_tensor((2, 2), device=device, dtype=torch.int64)
    f16 = make_tensor((2, 2), device=device, dtype=torch.float16)

    # int64 x int64 -- float32 result dtype
    result = traced_foo(i64, i64)
    assert result.dtype is torch.float32

    # int x int64 -- float32 result dtype
    result = traced_foo(2, i64)
    assert result.dtype is torch.float32

    # int64 x bool -- float32 result dtype
    result = traced_foo(i64, True)
    assert result.dtype is torch.float32

    # int64 x float16 -- float16 result dtype
    result = traced_foo(i64, f16)
    assert result.dtype is torch.float16


def test_atan2():
    def foo(a, b):
        return tlang.atan2(a, b)

    traced_foo = thunder.make_traced(foo, executor=executor_type)

    a = make_tensor((2, 2), device=device, dtype=torch.float32)
    b = make_tensor((2, 2), device=device, dtype=torch.float32)

    thunder_result = traced_foo(a, b)
    torch_result = torch.atan2(a, b)
    assert_close(thunder_result, torch_result)


def test_full():
    traced_full = thunder.make_traced(tlang.full, executor=executor_type)

    try:
        thunder_result = traced_full((1, 2, 3), 1.0, device=device, dtype=torch.float32)
    except Exception:
        pytest.skip("Expected to fail until connected to nvFuser full")

    torch_result = torch.full((1, 2, 3), 1.0, device=device, dtype=torch.float32)

    assert_close(thunder_result, torch_result)
