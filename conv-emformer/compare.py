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

    y = m(x, x_lens, num_processed_frames, states).squeeze(0)

    print("x", x.shape)
    print("y", y.shape)

    param = "m.ncnn.param"
    model = "m.ncnn.bin"
    with ncnn.Net() as net:
        net.opt.use_packing_layout = False
        net.load_param(param)
        net.load_model(model)

        with net.create_extractor() as ex:
            ex.input("in0", ncnn.Mat(x.squeeze(0).numpy()).clone())

            ret, ncnn_out0 = ex.extract("out0")

            ncnn_y = torch.from_numpy(ncnn_out0.numpy()).clone()
            print("y", y.shape, ncnn_y.shape)
            assert torch.allclose(y, ncnn_y, atol=1e-3), (y - ncnn_y).abs().max()


if __name__ == "__main__":
    torch.manual_seed(20221128)
    main()
