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

    def forward(self, x, x_lens, h, c):
        """
        Args:
          x:
            (N, T, C)
          x_lens:
            (N,)
          h:
            (num_layers, N, encoder_dim)
          c:
            (num_layers, N, rnn_hidden_size)
        """
        y = self.model(x, x_lens, h, c)
        #  y, (hx, cx) = self.lstm(x, (h, c))
        #  y = nn.functional.softmax(y, -1)
        return y


def main():
    f = Foo()
    convert_scaled_to_non_scaled(f, inplace=True, is_onnx=False)
    x = torch.zeros(1, 100, 80, dtype=torch.float32)
    x_lens = torch.tensor([100], dtype=torch.int64)
    states = f.model.get_init_states()

    m = torch.jit.trace(f, (x, x_lens, states[0], states[1]))
    print(m.graph)
    m.save("m.pt")


if __name__ == "__main__":
    torch.manual_seed(20230516)
    main()
