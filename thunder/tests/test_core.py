import pytest
import torch
from torch.testing import assert_close, make_tensor

import thunder
import thunder.core.lang as tlang
import thunder.langs.torch as ttorch

from .framework import executors, NOTHING

# TODO: move to test_elementwise by adding more numeric sample inputs
@executors(dtypes=(thunder.float32,))
def test_add_integer_constant(executor, device, dtype):
    def foo(a, b):
        c = tlang.add(a, 2)
        return tlang.add(c, b)

    traced_foo = thunder.make_traced(foo, executor=executor)

    tdtype = ttorch.torch_dtype(dtype)
    a = torch.testing.make_tensor((2, 1), device=device, dtype=tdtype)
    b = torch.testing.make_tensor((1, 2), device=device, dtype=tdtype)

    thunder_result = traced_foo(a, b)
    torch_result = (a + 2) + b

    torch.testing.assert_close(thunder_result, torch_result)

# TODO: move to test_elementwise by adding more numeric sample inputs
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

# TODO: move to test_elementwise by adding more numeric sample inputs
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

# TODO: move to test_elementwise by adding more numeric sample inputs
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

# TODO: move to test_elementwise by adding more numeric sample inputs
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


@executors(dtypes=(thunder.float32,))
def test_integer_isinstance_mimicry(executor, device, dtype):
    # isinstance() works as expected
    def foo(a, b, c):
        if isinstance(a, int):
            return tlang.add(a, b)

        return tlang.add(b, c)

    traced_foo = thunder.make_traced(foo, executor=executor)

    tdtype = ttorch.torch_dtype(dtype)
    a = make_tensor((2, 1), device=device, dtype=tdtype)
    b = make_tensor((2, 2), device=device, dtype=tdtype)
    c = make_tensor((1, 2), device=device, dtype=tdtype)

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

    traced_bar = thunder.make_traced(bar, executor=executor)

    try:
        thunder_result = traced_bar(a, b, c)
        torch_result = b + c
        assert_close(thunder_result, torch_result)
        pytest.fail()
    except BaseException:
        pass

    try:
        thunder_result = traced_bar(2, b, c)
        torch_result = 2 + b
        assert_close(thunder_result, torch_result)
        pytest.fail()
    except BaseException:
        pass


# FIXME NVIDIA: this test will cause a segmentation fault!
# def test_return_integer():
#     def foo(a, b):
#         return tlang.add(a, b)

#     traced_foo = thunder.make_traced(foo, executor=executor_type)

#     thunder_result = traced_foo(3, 4)
#     python_result = 3 + 4
#     assert_close(thunder_result, python_result)

# TODO: move to test_elementwise by adding more numeric sample inputs
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

# TODO: move to test_elementwise by adding more numeric sample inputs
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


# TODO: write these tests in test_prims.py (by first creating test_prim.py)
# def test_elementwise_binary_prim_shape_mismatch():
#     pass


# def test_elementwise_binary_prim_dtype_mismatch():
#     pass

# TODO: fix this test and put into test_reductions.py
@executors(dtypes=(thunder.float32,))
def test_torch_var(executor, device, dtype):
    # Tests passing all arguments as function inputs
    def foo(a, dim, *, keepdim=False, correction=1):
        return ttorch.var(a, dim, keepdim=keepdim, correction=correction)

    traced_foo = thunder.make_traced(foo, executor=executor)

    tdtype = ttorch.torch_dtype(dtype)
    a = torch.testing.make_tensor((4, 4), device=device, dtype=tdtype)

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

    traced_foo = thunder.make_traced(foo, executor=executor)

    a = torch.testing.make_tensor((4, 4), device=device, dtype=tdtype)

    thunder_result = traced_foo(a)
    torch_result = torch.var(a, [0, 1], keepdim=True, correction=2)
    assert_close(thunder_result, torch_result)

# TODO: put into test_reductions.py
@executors(dtypes=(thunder.float32,))
def test_torch_mean(executor, device, dtype):
    def foo(a, dim=None, keepdim=False, *, dtype=None):
        return ttorch.mean(a, dim, keepdim, dtype=dtype)

    traced_foo = thunder.make_traced(foo, executor=executor)

    tdtype = ttorch.torch_dtype(dtype)
    a = torch.testing.make_tensor((4, 4), device=device, dtype=tdtype)

    # Full reduction
    thunder_result = traced_foo(a, [0, 1])
    torch_result = torch.mean(a, [0, 1])
    assert_close(thunder_result, torch_result)

    # Reduce along dim 1
    thunder_result = traced_foo(a, [1])
    torch_result = torch.mean(a, [1])
    assert_close(thunder_result, torch_result)

# TODO: put into test_reductions.py
@executors(dtypes=(thunder.float32,))
def test_var_mean(executor, device, dtype):
    def foo(a, dim=None, unbiased=None, keepdim=False, *, correction=None):
        return ttorch.var_mean(a, dim, unbiased, keepdim=keepdim, correction=correction)

    traced_foo = thunder.make_traced(foo, executor=executor)

    tdtype = ttorch.torch_dtype(dtype)
    a = torch.testing.make_tensor((4, 4), device=device, dtype=tdtype)

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

    traced_foo = thunder.make_traced(foo, executor=executor)

    a = torch.testing.make_tensor((4, 4), device=device, dtype=tdtype)

    thunder_result = traced_foo(a)
    torch_result = torch.var_mean(a, [0, 1], keepdim=True, correction=2)
    assert_close(thunder_result, torch_result)

# TODO: test operator variants as part of opinfo-based tests
@executors(dtypes=(thunder.float32,))
def test_core_tensor_methods(executor, device, dtype):
    def foo(a, b, c, d):
        return a + b - c + (d - a)

    traced_foo = thunder.make_traced(foo, executor=executor)

    tdtype = ttorch.torch_dtype(dtype)
    a = torch.testing.make_tensor((4, 4), device=device, dtype=tdtype)
    b = torch.testing.make_tensor((2, 1, 4), device=device, dtype=tdtype)
    c = torch.testing.make_tensor((4, 1), device=device, dtype=tdtype)
    d = torch.testing.make_tensor((1, 1, 4), device=device, dtype=tdtype)

    thunder_result = traced_foo(a, b, c, d)
    torch_result = a + b - c + (d - a)
    assert_close(thunder_result, torch_result)


# TODO: this test just spot-checks type promotion -- it could probably be better
@executors(dtypes=NOTHING)
def test_type_promotion(executor, device, _):
    def foo(a, b):
        return a + b

    traced_foo = thunder.make_traced(foo, executor=executor)

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

    # b1 x int64 type promotion -- int64 result dtype
    result = traced_foo(b1, i64)
    assert result.dtype is torch.int64

    def bar(a, b, c):
        return a - b + c

    traced_bar = thunder.make_traced(bar, executor=executor)

    # float x int64 x float16 type promotion -- float16 result dtype
    result = traced_bar(2.0, i64, f16)
    assert result.dtype is torch.float16

    # float x int x int64 -- float32 result dtype
    result = traced_bar(2.1, -1, i64)
    assert result.dtype is torch.float32


@executors(dtypes=NOTHING)
def test_int_to_float_type_promotion(executor, device, _):
    def foo(a, b):
        return a / b

    traced_foo = thunder.make_traced(foo, executor=executor)

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

# TODO: put this in test_tensor_creation.py
@executors(dtypes=(thunder.float32,))
def test_full(executor, device, dtype):
    traced_full = thunder.make_traced(tlang.full, executor=executor)

    tdtype = ttorch.torch_dtype(dtype)
    try:
        thunder_result = traced_full((1, 2, 3), 1.0, device=device, dtype=tdtype)
    except Exception:
        pytest.skip("Expected to fail until connected to nvFuser full")

    torch_result = torch.full((1, 2, 3), 1.0, device=device, dtype=tdtype)

    assert_close(thunder_result, torch_result)


@executors(dtypes=(thunder.float32,))
def test_crazy_collections_in_and_out(executor, device, dtype):
    def foo(a, b, c, *, ka, kb, kc):
        d = {
            5: 2,
            7: 9,
            "a": [a, b],
            "b": {"a": a, "b": b, "c": [b, (a, c)]},
            "x": (a, [a, a, a], (b, (a, a, c, b))),
        }

        e = a['a']['a'] + b[0]
        f = c[1]['c'] + b[1]
        g = e + f
        h = f + ka + kb
        i = ka + ka  # NOTE: not returned (ignored computation)
        j = kc[0] + kc[1]

        return a, (g,), (((j,),),), g, g, b, e, (f, d, c, (d,), c, {"a": a, 5: f, 'b': h}), (5,), (), (a,), [5, a, (b,), (), {}], {}

    traced_foo = thunder.make_traced(foo, executor=executor)
    tdtype = ttorch.torch_dtype(dtype)

    a = make_tensor((2,), device=device, dtype=tdtype)
    b = make_tensor((2, 2, 2), device=device, dtype=tdtype)
    c = make_tensor((2, 2), device=device, dtype=tdtype)

    args = ({'a': {'a': a}}, (b, c), (3, {'c': c}))
    kwargs = {'ka': b, 'kb': 3., 'kc': (a, 2)}
    thunder_result = traced_foo(*args, **kwargs)
    torch_result = foo(*args, **kwargs)

    assert_close(thunder_result, torch_result)

@executors(dtypes=(thunder.float32,))
def test_no_return(executor, device, dtype):
    def foo(a, b):
        c = a + b
        pass

    traced_foo = thunder.make_traced(foo, executor=executor)
    tdtype = ttorch.torch_dtype(dtype)

    a = make_tensor((2,), device=device, dtype=tdtype)
    b = make_tensor((2, 2, 2), device=device, dtype=tdtype)

    thunder_result = traced_foo(a, b=b)
    torch_result = foo(a, b)

    assert_close(thunder_result, torch_result)

@executors(dtypes=NOTHING)
def test_no_input(executor, device, dtype):
    def foo():
        return 3, ()

    traced_foo = thunder.make_traced(foo, executor=executor)

    thunder_result = traced_foo()
    torch_result = foo()

    assert_close(thunder_result, torch_result)

@executors(dtypes=(thunder.float32,))
def test_no_compute(executor, device, dtype):
    def foo(a, b):
        return a, 3.

    traced_foo = thunder.make_traced(foo, executor=executor)
    tdtype = ttorch.torch_dtype(dtype)

    a = make_tensor((2,), device=device, dtype=tdtype)
    b = make_tensor((2, 2, 2), device=device, dtype=tdtype)

    thunder_result = traced_foo(a, b=b)
    torch_result = foo(a, b)

    assert_close(thunder_result, torch_result)
