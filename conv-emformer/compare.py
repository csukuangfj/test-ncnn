#!/usr/bin/env python3

#!/usr/bin/env python3

import ncnn
import torch

from generate import Foo
from scaling_converter import convert_scaled_to_non_scaled


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
    convert_scaled_to_non_scaled(f, inplace=True, is_onnx=False)

    m = torch.jit.load("m.pt")
    num_features = 80

    T = chunk_length + pad_length
    x = torch.rand(1, T, num_features)
    x_lens = torch.tensor([T])
    num_processed_frames = torch.tensor([0])
    states = f.encoder.init_states()

    y, y_lens, next_states = m(x, x_lens, num_processed_frames, states)
    y = y.squeeze(1)

    print("x", x.shape)
    print("y", y.shape)
    print("y_lens", y_lens)

    param = "m.ncnn.param"
    model = "m.ncnn.bin"
    with ncnn.Net() as net:
        net.opt.use_packing_layout = False
        #  net.opt.lightmode = False
        net.load_param(param)
        net.load_model(model)

        with net.create_extractor() as ex:
            ex.input("in0", ncnn.Mat(x.squeeze(0).numpy()).clone())

            x_lens = x_lens.float()
            ex.input("in1", ncnn.Mat(x_lens.numpy()).clone())

            ex.input("in2", ncnn.Mat(states[0].squeeze(1).numpy()).clone())

            #  num_processed_frames = num_processed_frames.float()
            #  ex.input("in2", ncnn.Mat(num_processed_frames.numpy()).clone())
            #  for i, s in enumerate(states):
            #      if i >= 4:
            #          break
            #      name = f"out{i+2}"
            #      print(name, states[i].shape)
            #      ex.input(name, ncnn.Mat(states[i].numpy()).clone())

            ret, ncnn_out0 = ex.extract("out0")
            ncnn_y = torch.from_numpy(ncnn_out0.numpy()).clone()

            print("y", y.shape, ncnn_y.shape)
            assert torch.allclose(y, ncnn_y, atol=1e-3), (y - ncnn_y).abs().max()

            ret, ncnn_out1 = ex.extract("out1")
            ncnn_y_lens = torch.from_numpy(ncnn_out1.numpy()).clone().long()

            print("y_lens", ncnn_y_lens, y_lens)
            assert torch.allclose(y_lens, ncnn_y_lens, atol=1e-3), (
                (y_lens - ncnn_y_lens).abs().max()
            )


if __name__ == "__main__":
    torch.manual_seed(20221128)
    main()
