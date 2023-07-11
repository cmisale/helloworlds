import os
import sys
import torch

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rank = os.environ["RANK"]
    size = os.environ["WORLD_SIZE"]
    node = os.environ["NODE"]

    sys.stdout.write(
    "Hope this code finds you well. Process %s of %s on %s. Device %s. GPU count %d\n"
    % (rank, size, node, device, torch.cuda.device_count()))
