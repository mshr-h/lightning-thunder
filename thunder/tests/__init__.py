from thunder.executors import TORCH_AVAILABLE, NVFUSER_AVAILABLE

if NVFUSER_AVAILABLE:
    executor_type = "nvfuser"
elif TORCH_AVAILABLE:
    executor_type = "torch"
else:
    RuntimeError("No executors can be selected. Either NVFuser or Torch must be available.")

if executor_type == "nvfuser":
    supported_device_types = ("cuda",)
    device = "cuda"
elif executor_type == "torch":
    import torch
    if torch.has_cuda:
        supported_device_types = ("cpu", "cuda")
        device = "cuda"
    else:
        supported_device_types = ("cpu",)
        device = "cpu"
