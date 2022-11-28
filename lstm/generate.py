#!/usr/bin/env python3

import torch
import torch.nn as nn


class Foo(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        proj_size: int,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            proj_size=proj_size,
            num_layers=1,
            batch_first=False,
        )

    def forward(self, x, h, c):
        """
        Args:
          x:
            (T, N, input_size)
          h:
            (num_layers, N, proj_size or hidden_size)
          c:
            (num_layers, N, hidden_size)
        """
        y, (hx, cx) = self.lstm(x, (h, c))
        y = nn.functional.softmax(y, -1)
        return y, hx, cx


def main():
    input_size = 512
    hidden_size = 1024
    proj_size = 768
    f = Foo(
        input_size=input_size,
        hidden_size=hidden_size,
        proj_size=proj_size,
    )

    N = 1
    T = 5
    x = torch.rand(T, N, input_size)
    h0 = torch.rand(1, N, proj_size or hidden_size)
    c0 = torch.rand(1, N, hidden_size)
    m = torch.jit.trace(f, (x, h0, c0))
    print(m.graph)
    m.save("m.pt")


if __name__ == "__main__":
    torch.manual_seed(20220924)
    main()
