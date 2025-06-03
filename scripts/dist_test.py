# dist_test.py
import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

# Print environment to confirm LOCAL_RANK mapping
print(
    "ENV:",
    {
        "WORLD_SIZE": os.environ.get("WORLD_SIZE"),
        "RANK": os.environ.get("RANK"),
        "LOCAL_RANK": os.environ.get("LOCAL_RANK"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
    },
)


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)


def setup_process_group(backend: str):
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    print(f"rank: {rank}, world_size: {world_size}, backend: {backend}")
    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
    )
    torch.manual_seed(42)


def demo_ddp():
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    backend = os.environ.get("BACKEND", "nccl")

    print(
        f"Starting demo_ddp | rank: {rank}, local_rank: {local_rank}, world_size: {world_size}"
    )

    # Bind to GPU if using NCCL
    if backend == "nccl":
        torch.cuda.set_device(local_rank)  # local_rank < visible GPUs
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    setup_process_group(backend)

    model = SimpleModel().to(device)
    ddp_model = DDP(model, device_ids=[local_rank] if backend == "nccl" else None)

    # Dummy data and optimizer
    x = torch.randn(4, 10).to(device)
    y = torch.randn(4, 10).to(device)
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(x)
    loss = nn.functional.mse_loss(outputs, y)
    loss.backward()  # AllReduce on gradients  [oai_citation:24‡GitHub](https://github.com/Lightning-AI/pytorch-lightning/discussions/8630?utm_source=chatgpt.com) [oai_citation:25‡PyTorch Forums](https://discuss.pytorch.org/t/emulate-distributed-training-setup-with-1-gpu/199084?utm_source=chatgpt.com)
    optimizer.step()

    # Verify collective communication
    tensor = torch.tensor([rank], device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    if rank == 0:
        print(f"All-reduced sum: {tensor.item()} (expected {sum(range(world_size))})")

    dist.destroy_process_group()


if __name__ == "__main__":
    demo_ddp()
