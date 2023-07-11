import os
import sys
import torch
import torch.distributed as dist

if __name__ == "__main__":
    print("Torch distributed available: ", torch.distributed.is_available())
    if torch.cuda.is_available():
        dist.init_process_group("nccl")
        device = torch.device("cuda")
    else:
        dist.init_process_group("gloo")
        device = torch.device("cpu")

    rank = os.environ["RANK"]
    size = os.environ["WORLD_SIZE"]
    node = os.environ["NODE"]

    sys.stdout.write(
    "Hope this code finds you well. Process %s of %s on %s. Device: %s. GPU count: %d\n"
    % (rank, size, node, device, torch.cuda.device_count()))
