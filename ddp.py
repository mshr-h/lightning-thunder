import os
import torch
import thunder
from torch import nn


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


rank = 0
world_size = 1
device = torch.device(f"cuda:{rank}")

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12345"
torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)

model = ToyModel().to(device)
ddp_model = thunder.distributed.ddp(model)
jitted_model = thunder.jit(ddp_model)

x = torch.full((10,), 1, dtype=torch.float32, device=device)
y = jitted_model(x)
print(y)

torch.distributed.destroy_process_group()
