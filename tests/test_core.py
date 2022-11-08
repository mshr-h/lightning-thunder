import pytest

import torch
import thunder
import thunder.core.lang as tlang
import thunder.core.proxies as proxies

from torch.testing import make_tensor, assert_close


# TODO: add device/dtype instantiation
# TODO: use OpInfo samples
def test_add():
    def foo(a, b):
        return tlang.add(a, b)

    traced_foo = thunder.make_traced(foo)

    a = torch.testing.make_tensor((2, 2), device="cuda", dtype=torch.float32)
    b = torch.testing.make_tensor((2, 2), device="cuda", dtype=torch.float32)

    thunder_result = traced_foo(a, b)
    torch_result = a + b

    torch.testing.assert_close(thunder_result, torch_result)


def test_add_broadcast():
    def foo(a, b):
        return tlang.add(a, b)

    traced_foo = thunder.make_traced(foo)

    a = torch.testing.make_tensor((2, 1), device="cuda", dtype=torch.float32)
    b = torch.testing.make_tensor((1, 2), device="cuda", dtype=torch.float32)

    thunder_result = traced_foo(a, b)
    torch_result = a + b

    torch.testing.assert_close(thunder_result, torch_result)


def test_add_integer_constant():
    def foo(a, b):
        c = tlang.add(a, 2)
        return tlang.add(c, b)

    traced_foo = thunder.make_traced(foo)

    a = torch.testing.make_tensor((2, 1), device="cuda", dtype=torch.float32)
    b = torch.testing.make_tensor((1, 2), device="cuda", dtype=torch.float32)

    thunder_result = traced_foo(a, b)
    torch_result = (a + 2) + b

    torch.testing.assert_close(thunder_result, torch_result)


def test_add_integer_input():
    def foo(a, b):
        return tlang.add(a, b)

    traced_foo = thunder.make_traced(foo)

    a = make_tensor((2, 1), device="cuda", dtype=torch.float32)

    thunder_result = traced_foo(a, 3)
    torch_result = a + 3

    assert_close(thunder_result, torch_result)


def test_add_integer_inputs():
    def foo(a, b, c):
        d = tlang.add(a, b)
        return tlang.add(c, d)

    traced_foo = thunder.make_traced(foo)

    a = make_tensor((3, 2), device="cuda", dtype=torch.float32)

    thunder_result = traced_foo(3, 4, a)
    torch_result = 3 + 4 + a
    assert_close(thunder_result, torch_result)


def test_add_integer_constants():
    def foo(a):
        b = tlang.add(2, 3)
        return tlang.add(a, b)

    traced_foo = thunder.make_traced(foo)

    a = make_tensor((2, 4), device="cuda", dtype=torch.float32)

    thunder_result = traced_foo(a)
    torch_result = 5 + a
    assert_close(thunder_result, torch_result)


def test_add_floats():
    def foo(a, b):
        c = tlang.add(2.0, a)
        return tlang.add(b, c)

    traced_foo = thunder.make_traced(foo)

    a = make_tensor((2, 4), device="cuda", dtype=torch.float32)

    thunder_result = traced_foo(0.7, a)
    torch_result = 2.0 + 0.7 + a
    assert_close(thunder_result, torch_result)


def test_integer_isinstance_mimicry():
    def foo(a, b):
        if isinstance(a, int):
            return tlang.add(a, b)

        return b

    traced_foo = thunder.make_traced(foo)

    a = make_tensor((2, 1), device="cuda", dtype=torch.float32)
    b = make_tensor((2, 2), device="cuda", dtype=torch.float32)

    # FIXME: this case will throw
    #   RuntimeError: entry_it != disjointSetMap().end() INTERNAL ASSERT FAILED at "../torch/csrc/jit/codegen/cuda/disjoint_set.h":259, please report a bug to PyTorch. Strict mapping failed on element: T1_g[ iS2{i4}, iS3{i5} ] either an error occured, or non strict mapping should have been used.
    #   Possibly because the trace is just inputs and an output?
    try:
        thunder_result = traced_foo(a, b)
        torch_result = b
        assert_close(thunder_result, torch_result)
        pytest.fail()
    except:
        pass

    thunder_result = traced_foo(2, b)
    torch_result = 2 + b
    assert_close(thunder_result, torch_result)


# FIXME NVIDIA: this test will cause a segmentation fault!
# def test_return_integer():
#     def foo(a, b):
#         return tlang.add(a, b)

#     traced_foo = thunder.make_traced(foo)

#     thunder_result = traced_foo(3, 4)
#     python_result = 3 + 4
#     assert_close(thunder_result, python_result)


def test_abs():
    def foo(a):
        return tlang.abs(a)

    traced_foo = thunder.make_traced(foo)

    a = make_tensor((2, 8), device="cuda", dtype=torch.float32)

    thunder_result = traced_foo(a)
    torch_result = torch.abs(a)
    assert_close(thunder_result, torch_result)


def test_abs_integer():
    def foo(a, b):
        a_abs = tlang.abs(a)
        return tlang.add(a_abs, b)

    traced_foo = thunder.make_traced(foo)

    a = -3
    b = make_tensor((1, 8), device="cuda", dtype=torch.float32)

    thunder_result = traced_foo(a, b)
    torch_result = 3 + b
    assert_close(thunder_result, torch_result)


def test_abs_float():
    def foo(a, b):
        a_abs = tlang.abs(a)
        return tlang.add(a_abs, b)

    traced_foo = thunder.make_traced(foo)

    a = -2.7
    b = make_tensor((1, 8), device="cuda", dtype=torch.float32)

    thunder_result = traced_foo(a, b)
    torch_result = abs(a) + b
    assert_close(thunder_result, torch_result)


def test_elementwise_binary_prim_shape_mismatch():
    pass


def test_elementwise_binary_prim_dtype_mismatch():
    pass
