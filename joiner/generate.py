#!/usr/bin/env python3

import torch
import torch.nn as nn
from joiner import Joiner
from scaling_converter import convert_scaled_to_non_scaled


class Foo(nn.Module):
    def __init__(
        self,
        encoder_dim: int = 512,
        decoder_dim: int = 512,
        joiner_dim: int = 512,
        vocab_size: int = 500,
    ):
        super().__init__()

        self.joiner = Joiner(
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim,
            joiner_dim=joiner_dim,
            vocab_size=vocab_size,
        )

    def forward(self, encoder_out, decoder_out):
        """
        Args:
          encoder_out:
            A tensor of shape (N, encoder_dim)
          decoder_out:
            A tensor of shape (N, decoder_dim)
        Returns:
          Return a tensor of shape (N, vocab_size)
          Note: There is no softmax or log-softmax.
        """
        y = self.joiner(encoder_out, decoder_out, project_input=True)
        return y


@torch.no_grad()
def main():
    encoder_dim = 512
    decoder_dim = 512
    vocab_size = 500
    f = Foo(encoder_dim=encoder_dim, decoder_dim=decoder_dim, vocab_size=vocab_size)
    f.eval()

    convert_scaled_to_non_scaled(f, inplace=True, is_onnx=False)
    encoder_out = torch.rand(1, encoder_dim)
    decoder_out = torch.rand(1, decoder_dim)
    m = torch.jit.trace(f, (encoder_out, decoder_out))
    print(m.graph)
    m.save("m.pt")


if __name__ == "__main__":
    torch.manual_seed(20221129)
    main()
