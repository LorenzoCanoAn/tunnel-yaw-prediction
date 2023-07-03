from tunnel_yaw_prediction.models import TunnelYawPredictor
import torch
from time import time_ns as ns
from torchsummary import summary


def main():
    sizes = (30, 60, 100, 200, 300, 400)
    n_runs_forward = 50000
    for size in sizes:
        model = TunnelYawPredictor()
        model.eval()
        model.to("cpu")
        start = ns()
        for _ in range(n_runs_forward):
            model.forward(torch.ones((1, 1, size, size)))
        end = ns()
        elapsed = (end - start) * 1e-9
        time_per_run = elapsed / n_runs_forward
        frequency = 1 / time_per_run
        print(
            f"for size ({size:4d},{size:4d}) took {elapsed:010f} for {n_runs_forward} runs: time_per_run: {time_per_run:010f} -> frequency: {frequency:010f}"
        )


if __name__ == "__main__":
    main()
