NVFUSER_AVAILABLE = True
try:
    import torch._C._nvfuser
except ModuleNotFoundError:
    NVFUSER_AVAILABLE = False
