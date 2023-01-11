import sys

import pytest
import torch
from torch.testing import assert_close

import thunder.core.script.frontend
import thunder.core.script.python_ir
import thunder.core.script.passes
from torch import add as tadd


def sample_add_fn(x, y):
    return tadd(x, y)


class M1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(3, 5)
        self.b = torch.nn.Linear(5, 4)

    def forward(self, x: torch.Tensor, flag: bool = True):
        # while flag:
        #    x = 2 * x
        if flag:
            return self.a(x)
        return 2 * x


@pytest.mark.skipif(
    sys.version_info < (3, 10) or sys.version_info >= (3, 11),
    reason="requires python3.10",
)
def test_acquisition_compile():
    model = M1()
    gr = thunder.core.script.frontend.acquire_method(model.forward)

    # TODO (t-vi): should these be called automatically? yes.
    thunder.core.script.frontend.make_ssa(gr)
    thunder.core.script.frontend.make_single_return(gr)
    fn = thunder.core.script.python_ir.generate_function(gr)

    a = torch.randn(2, 3)
    assert_close(model(a, True), fn(model, a, True))
    assert_close(model(a, False), fn(model, a, False))


@pytest.mark.skipif(
    sys.version_info < (3, 10) or sys.version_info >= (3, 11),
    reason="requires python3.10",
)
def test_torch_to_thunder():
    gr = thunder.core.script.frontend.acquire_method(sample_add_fn)
    thunder.core.script.frontend.make_ssa(gr)
    thunder.core.script.frontend.make_single_return(gr)
    thunder.core.script.passes.torch_to_thunder(gr)
    thunder_fn = thunder.core.script.python_ir.generate_function(gr)

    traced_fn = thunder.make_traced(thunder_fn, executor="torch")
    a = torch.randn((2, 2), device="cpu", dtype=torch.float32)
    b = torch.randn((2, 2), device="cpu", dtype=torch.float32)

    res = traced_fn(a, b)
    expected = sample_add_fn(a, b)
    assert_close(res, expected)


def test_sequential():
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 5),
        torch.nn.Tanh(),
        torch.nn.Linear(5, 3),
    )

    gr = thunder.core.script.frontend.acquire_method(model.forward)
    thunder.core.script.frontend.make_ssa(gr)
    thunder.core.script.frontend.make_single_return(gr)
    fn = thunder.core.script.python_ir.generate_function(gr)

    a = torch.randn(2, 3)
    assert_close(model(a), fn(model, a))
