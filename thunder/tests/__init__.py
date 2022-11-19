import torch

from thunder.executors import NVFUSER_AVAILABLE

executor_type = "nvfuser" if NVFUSER_AVAILABLE else "torch"

if executor_type == "nvfuser":
    supported_device_types = ("cuda",)
elif executor_type == "torch":
    if torch.has_cuda:
        supported_device_types = ("cpu", "cuda")
    else:
        supported_device_types = ("cpu",)

device = "cuda" if torch.has_cuda else "cpu"
