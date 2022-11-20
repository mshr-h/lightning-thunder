import os

from thunder.executors import TORCH_AVAILABLE, NVFUSER_AVAILABLE

executor_env_var = os.environ.get("THUNDER_EXECUTOR")

if executor_env_var:
    executor_type = executor_env_var
else:
    executor_type = "nvfuser" if NVFUSER_AVAILABLE else "torch"

if executor_type == "nvfuser" and not NVFUSER_AVAILABLE:
    RuntimeError("NVFuser executor requested but NVFuser is not available")
elif executor_type == "torch" and not TORCH_AVAILABLE:
    RuntimeError("Torch executor requested but Torch is not available")
else:
    RuntimeError(f"Unsupported executor type {executor_type}")

if executor_type == "nvfuser":
    supported_device_types = ("cuda",)
elif executor_type == "torch":
    import torch
    if torch.has_cuda:
        supported_device_types = ("cpu", "cuda")
    else:
        supported_device_types = ("cpu",)
