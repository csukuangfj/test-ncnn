#!/usr/bin/env python3

import torch
import torch.nn as nn
from emformer import Emformer
from scaling_converter import convert_scaled_to_non_scaled
from emformer import stack_states
from typing import List


class Foo(nn.Module):
    def __init__(
        self,
        num_features: int = 80,
        chunk_length: int = 32,
        d_model: int = 512,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        layer_dropout: float = 0.075,
        cnn_module_kernel: int = 31,
        num_encoder_layers: int = 12,
        left_context_length: int = 32,
        right_context_length: int = 8,
        memory_size: int = 32,
        tanh_on_mem: bool = False,
        negative_inf: float = -1e8,
        is_pnnx: bool = True,
    ):
        super().__init__()
        self.encoder = Emformer(
            num_features=num_features,
            chunk_length=chunk_length,
            subsampling_factor=4,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_encoder_layers=num_encoder_layers,
            cnn_module_kernel=cnn_module_kernel,
            left_context_length=left_context_length,
            right_context_length=right_context_length,
            memory_size=memory_size,
            is_pnnx=is_pnnx,
        )

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        num_processed_frames: torch.Tensor,
        states: List[torch.Tensor],
    ):
        """
        Args:
          x:
            (N, T, num_features)
          x_lens:
            (N,)
        """
        y = self.encoder.infer(x, x_lens, num_processed_frames, states)
        return y


@torch.no_grad()
def main():
    num_features = 80
    d_model = 512
    chunk_length = 32  # before subsampling
    right_context_length = 8  # before subsampling
    pad_length = 8 + 2 * 4 + 3

    f = Foo(
        num_features=num_features,
        d_model=d_model,
        chunk_length=chunk_length,
        right_context_length=right_context_length,
    )
    f.eval()

    N = 1

    T = chunk_length + pad_length
    x = torch.rand(N, T, num_features)
    x_lens = torch.tensor([T])
    num_processed_frames = torch.tensor([0])
    states = f.encoder.init_states()
    convert_scaled_to_non_scaled(f, inplace=True, is_onnx=False)
    y, y_lens, next_states = f(x, x_lens, num_processed_frames, states)
    print("x", x.shape)
    print("y", y.shape)
    print("y_lens", y_lens)
    m = torch.jit.trace(f, (x, x_lens, num_processed_frames, states))
    #  print(m.graph)
    m.save("m.pt")


if __name__ == "__main__":
    torch.manual_seed(20221128)
    main()
