#!/usr/bin/env python3

from typing import List

import torch
import torch.nn as nn
from scaling_converter import convert_scaled_to_non_scaled

from zipformer import Zipformer


def get_encoder_model() -> nn.Module:
    def to_int_tuple(s: str):
        return tuple(map(int, s.split(",")))

    encoder = Zipformer(
        num_features=80,
        output_downsampling_factor=2,
        zipformer_downsampling_factors=to_int_tuple("1,2,4,8,2"),
        encoder_dims=to_int_tuple("384,384,384,384,384"),
        attention_dim=to_int_tuple("192,192,192,192,192"),
        encoder_unmasked_dims=to_int_tuple("256,256,256,256,256"),
        nhead=to_int_tuple("8,8,8,8,8"),
        feedforward_dim=to_int_tuple("1024,1024,2048,2048,1024"),
        cnn_module_kernels=to_int_tuple("31,31,31,31,31"),
        num_encoder_layers=to_int_tuple("2,4,3,2,4"),
        num_left_chunks=4,
        short_chunk_size=50,
        decode_chunk_size=(32 // 2),
        is_pnnx=True,
    )

    # Here, we replace forward with streaming_forward
    encoder.__class__.forward = encoder.__class__.streaming_forward
    return encoder


class Foo(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.encoder = get_encoder_model()

    def forward(
        self, x: torch.Tensor, x_lens: torch.Tensor, states: List[torch.Tensor]
    ):
        """
        Args:
          x:
            (N, T, C)
          x_lens:
            (N,)
        """
        return self.encoder(x, x_lens, states)


@torch.no_grad()
def main():
    f = Foo()
    f.eval()

    convert_scaled_to_non_scaled(f, inplace=True)

    decode_chunk_len = f.encoder.decode_chunk_size * 2

    pad_length = 7
    T = decode_chunk_len + pad_length

    states = f.encoder.get_init_state()

    x = torch.zeros(1, T, 80, dtype=torch.float32)
    x_lens = torch.full((1,), T, dtype=torch.int32)

    m = torch.jit.trace(f, (x, x_lens, states))
    #  print(m.graph)
    print(m.inlined_graph)
    m.save("m.pt")


if __name__ == "__main__":
    torch.manual_seed(20230115)
    main()
