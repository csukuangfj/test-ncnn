#!/usr/bin/env python3

import torch
import torch.nn as nn
from decoder import Decoder
from scaling_converter import convert_scaled_to_non_scaled


class Foo(nn.Module):
    def __init__(
        self,
        vocab_size: int = 500,
        decoder_dim: int = 512,
        blank_id: int = 0,
        context_size: int = 2,
    ):
        super().__init__()

        self.decoder = Decoder(
            vocab_size=vocab_size,
            decoder_dim=decoder_dim,
            blank_id=blank_id,
            context_size=context_size,
        )

    def forward(self, x):
        """
        Args:
          x:
            (N, context_size). Note: It has to be of type torch.int32 for ncnn
        """
        y = self.decoder(x, need_pad=False)
        return y


@torch.no_grad()
def main():
    context_size = 2
    vocab_size = 500
    f = Foo(context_size=context_size, vocab_size=500)
    f.eval()
    N = 1
    T = context_size
    convert_scaled_to_non_scaled(f, inplace=True, is_onnx=False)
    x = torch.randint(low=0, high=vocab_size, size=(N, T))
    m = torch.jit.trace(f, x)
    print(m.graph)
    m.save("m.pt")


if __name__ == "__main__":
    torch.manual_seed(20221129)
    main()
