import os

# get value of environment variable, so that the executor
# can be forced when running tests
# THUNDER_EXECUTOR=nvfuser pytest
executor_env_var = os.environ.get("THUNDER_EXECUTOR")

# evaluate whether torch is available
TORCH_AVAILABLE = True
try:
    import torch
except ModuleNotFoundError:
    TORCH_AVAILABLE = False

# evaluate whether nvfuser is available
NVFUSER_AVAILABLE = True
try:
    import torch._C._nvfuser
except ModuleNotFoundError:
    NVFUSER_AVAILABLE = False

if executor_env_var:
    # if an executor was specified through the env var, then choose that
    executor_type = executor_env_var
else:
    # otherwise, choose nvfuser if available, else choose torch
    # this allows to default to nvfuser when running on a CUDA machine
    # and torch otherwise
    executor_type = "nvfuser" if NVFUSER_AVAILABLE else "torch"

if executor_type == "nvfuser" and not NVFUSER_AVAILABLE:
    RuntimeError("NVFuser executor requested but NVFuser is not available")
elif executor_type == "torch" and not TORCH_AVAILABLE:
    RuntimeError("Torch executor requested but Torch is not available")
else:
    RuntimeError(f"Unsupported executor type {executor_type}")

if executor_type == "nvfuser":
    # NVFuser only supports CUDA devices
    supported_device_types = ("cuda",)
elif executor_type == "torch":
    import torch
    if torch.has_cuda:
        # Torch supports both CPU and CUDA on a GPU machine
        supported_device_types = ("cpu", "cuda")
    else:
        # Only support CPU on a non-GPU machine
        supported_device_types = ("cpu",)
