"""A graphics load for the utilization tests: matrix multiplications on the ROCm device until standard input closes.

Runs in the workshop's own environment (torch with ROCm). Prints "running" after the first multiplication.
"""

from __future__ import annotations

import select
import sys

import torch


def main() -> None:
    matrix = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    torch.cuda.synchronize()
    print("running", flush=True)
    count = 0
    while True:
        readable, _, _ = select.select([sys.stdin], [], [], 0)
        if readable and sys.stdin.readline() == "":
            break
        matrix = matrix @ matrix * 1e-3
        torch.cuda.synchronize()
        count += 1
    print(f"stopped {count}", flush=True)


if __name__ == "__main__":
    main()
