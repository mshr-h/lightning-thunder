import sys

import pytest
import torch
from torch import add as tadd
from torch.testing import assert_close

import thunder.core.script.frontend
import thunder.core.script.passes
import thunder.core.script.python_ir

from . import nanogpt_model


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


@pytest.mark.skipif(
    sys.version_info < (3, 10) or sys.version_info >= (3, 11),
    reason="requires python3.10",
)
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


@pytest.mark.skipif(
    sys.version_info < (3, 10) or sys.version_info >= (3, 11),
    reason="requires python3.10",
)
def test_nanogpt_basic():
    model = nanogpt_model.GPT(nanogpt_model.GPTConfig)

    gr = thunder.core.script.frontend.acquire_method(model.forward)
    thunder.core.script.frontend.make_ssa(gr)
    thunder.core.script.frontend.make_single_return(gr)
    fn = thunder.core.script.python_ir.generate_function(gr)

    x = torch.randint(0, 255, (5, 5))
    torch.manual_seed(5)
    res, _ = fn(model, x, None)
    torch.manual_seed(5)
    expected, _ = model.forward(x)

    assert_close(res, expected)


@pytest.mark.skipif(
    sys.version_info < (3, 10) or sys.version_info >= (3, 11),
    reason="requires python3.10",
)
def test_split_block():
    def foo(a, b):
        c = a + b
        d = a + c
        return d

    gr = thunder.core.script.frontend.acquire_method(foo, verbose=False)
    thunder.core.script.frontend.make_ssa(gr)
    thunder.core.script.frontend.make_single_return(gr)
    thunder.core.script.passes.split_block(gr, gr.blocks[0], gr.blocks[0].nodes[1])
    dot = thunder.core.script.graph.make_dot(gr, add_names=True)
    fn = thunder.core.script.python_ir.generate_function(gr)

    a = torch.randn(5)
    b = torch.randn(5)
    assert_close(fn(a, b), foo(a, b))


@pytest.mark.skipif(
    sys.version_info < (3, 10) or sys.version_info >= (3, 11),
    reason="requires python3.10",
)
def test_inline_submodule():
    class MLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = torch.nn.Linear(5, 10)
            self.l2 = torch.nn.Linear(10, 5)

        def forward(self, x):
            return self.l2(torch.tanh(self.l1(x)))

    m = MLP()
    gr = thunder.core.script.frontend.acquire_method(m.forward, verbose=False)
    thunder.core.script.frontend.make_ssa(gr)
    thunder.core.script.frontend.make_single_return(gr)

    nodes_to_inline = [gr.blocks[0].nodes[0], gr.blocks[0].nodes[2]]
    for n in nodes_to_inline:
        thunder.core.script.passes.inline_method_call(gr, n)

    assert len(gr.blocks) > 1

    thunder.core.script.passes.merge_blocks_where_possible(gr)

    assert len(gr.blocks) == 1

    fn = thunder.core.script.python_ir.generate_function(gr)

    x = torch.randn(5, 5)
    assert_close(fn(m, x), m(x))

    # explicitly check for things to have been inlined?


@pytest.mark.skipif(
    sys.version_info < (3, 10) or sys.version_info >= (3, 11),
    reason="requires python3.10",
)
def test_inline_submodule_and_convert_to_thunder():
    model = nanogpt_model.MLP(nanogpt_model.GPTConfig)

    gr = thunder.core.script.frontend.acquire_method(model.forward, verbose=False)
    thunder.core.script.frontend.make_ssa(gr)
    thunder.core.script.frontend.make_single_return(gr)

    thunder.core.script.passes.inline_submodule_calls(gr)
    thunder.core.script.passes.merge_blocks_where_possible(gr)
    thunder.core.script.passes.torch_to_thunder(gr)

    fn = thunder.core.script.python_ir.generate_function(gr)

    ### now trace fn and check things work...
