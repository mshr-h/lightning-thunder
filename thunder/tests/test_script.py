import sys

import pytest
import torch
from torch.testing import assert_close

import thunder.core.script.frontend
import thunder.core.script.python_ir


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
    print(gr)
    fn = thunder.core.script.python_ir.generate_function(gr)

    a = torch.randn(2, 3)
    assert_close(model(a, True), fn(model, a, True))
    assert_close(model(a, False), fn(model, a, False))
