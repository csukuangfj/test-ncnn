#!/usr/bin/env python3

import torch
import torch.nn as nn
from lstm import RNN
from scaling_converter import convert_scaled_to_non_scaled


class Foo(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.model = RNN(
            num_features=80,
            subsampling_factor=4,
            d_model=512,
            rnn_hidden_size=1024,
            dim_feedforward=2048,
            num_encoder_layers=12,
            aux_layer_period=0,
            is_pnnx=True,
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
        #  y, (hx, cx) = self.lstm(x, (h, c))
        #  y = nn.functional.softmax(y, -1)
        return y, h, c


def main():
    f = Foo()
    convert_scaled_to_non_scaled(f, inplace=True, is_onnx=False)

    print(f)
    return

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
