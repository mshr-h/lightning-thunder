TORCH_AVAILABLE = True
try:
    import torch
except ModuleNotFoundError:
    TORCH_AVAILABLE = False


NVFUSER_AVAILABLE = True
try:
    import torch._C._nvfuser
except ModuleNotFoundError:
    NVFUSER_AVAILABLE = False
