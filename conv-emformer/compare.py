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
    f = Foo(
        num_features=num_features,
        d_model=d_model,
    )
    convert_scaled_to_non_scaled(f, inplace=True, is_onnx=False)

    m = torch.jit.load("m.pt")
    num_features = 80

    T = 30
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
        net.load_param(param)
        net.load_model(model)

        with net.create_extractor() as ex:
            ex.input("in0", ncnn.Mat(x.squeeze(0).numpy()).clone())
            x_lens = x_lens.float()
            ex.input("in1", ncnn.Mat(x_lens.numpy()).clone())

            ret, ncnn_out0 = ex.extract("out0")
            ret, ncnn_out1 = ex.extract("out1")

            ncnn_y = torch.from_numpy(ncnn_out0.numpy()).clone()
            ncnn_y_lens = torch.from_numpy(ncnn_out1.numpy()).clone().long()

            print("y", y.shape, ncnn_y.shape)
            assert torch.allclose(y, ncnn_y, atol=1e-3), (y - ncnn_y).abs().max()

            print("y_lens", ncnn_y_lens, y_lens)
            assert torch.allclose(y_lens, ncnn_y_lens, atol=1e-3), (
                (y_lens - ncnn_y_lens).abs().max()
            )


if __name__ == "__main__":
    torch.manual_seed(20221128)
    main()
