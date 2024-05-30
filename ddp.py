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


if __name__ == "__main__":
    rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
    world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)

    device = torch.device(f"cuda:{rank}")
    model = ToyModel().to(device)
    ddp_model = thunder.distributed.ddp(model)
    jitted_model = thunder.jit(ddp_model)

    x = torch.full((10,), 1, dtype=torch.float32, device=device)
    y = jitted_model(x)
    print(y)

    torch.distributed.destroy_process_group()
