import torch
import thunder
import thunder.core.lang as tlang


# TODO: add device/dtype instantiation
# TODO: use OpInfo samples
def test_add():

    def foo(a, b):
        return tlang.add(a, b)
    
    traced_foo = thunder.make_traced(foo)

    a = torch.testing.make_tensor((2, 2), device='cuda', dtype=torch.float32)
    b = torch.testing.make_tensor((2, 2), device='cuda', dtype=torch.float32)

    thunder_result = traced_foo(a, b)
    torch_result = a + b

    torch.testing.assert_close(thunder_result, torch_result)

def test_add_broadcast():

    def foo(a, b):
        return tlang.add(a, b)
    
    traced_foo = thunder.make_traced(foo)

    a = torch.testing.make_tensor((2, 1), device='cuda', dtype=torch.float32)
    b = torch.testing.make_tensor((1, 2), device='cuda', dtype=torch.float32)

    thunder_result = traced_foo(a, b)
    torch_result = a + b

    torch.testing.assert_close(thunder_result, torch_result)