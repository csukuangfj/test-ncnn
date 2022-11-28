#!/usr/bin/env python3

import torch
import torch.nn as nn
from lstm import Conv2dSubsampling
from scaling_converter import convert_scaled_to_non_scaled


class Foo(nn.Module):
    def __init__(
        self,
        num_features: int = 80,
        d_model: int = 512,
        is_pnnx: bool = True,
    ):
        super().__init__()

        self.encoder_embed = Conv2dSubsampling(
            num_features,
            d_model,
            is_pnnx=is_pnnx,
        )

    def forward(self, x):
        """
        Args:
          x:
            (N, T, num_features)
        """
        y = self.encoder_embed(x)
        return y


def main():
    num_features = 80
    d_model = 512
    f = Foo(
        num_features=num_features,
        d_model=d_model,
    )

    N = 1
    T = 20
    x = torch.rand(N, T, num_features)
    convert_scaled_to_non_scaled(f, inplace=True, is_onnx=False)
    m = torch.jit.trace(f, x)
    print(m.graph)
    m.save("m.pt")


if __name__ == "__main__":
    torch.manual_seed(20220924)
    main()
