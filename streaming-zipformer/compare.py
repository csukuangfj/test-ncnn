#!/usr/bin/env python3

#!/usr/bin/env python3

import ncnn
import torch

from generate import Foo
from scaling_converter import convert_scaled_to_non_scaled


@torch.no_grad()
def main():
    f = Foo()
    f.eval()
    convert_scaled_to_non_scaled(f, inplace=True)

    decode_chunk_len = f.encoder.decode_chunk_size * 2
    pad_length = 7
    T = decode_chunk_len + pad_length

    m = torch.jit.load("m.pt")
    num_features = 80

    x = torch.rand(1, T, num_features)
    x_lens = torch.tensor([T])
    states = f.encoder.get_init_state()

    y = m(x, x_lens, states)

    print((x.shape[1] - 7) // 2)

    print("x", x.shape)  # (1, 39, 80)
    print("y", y.shape)  # (1, 16, 384)

    y = y.squeeze(0)

    param = "m.ncnn.param"
    model = "m.ncnn.bin"
    with ncnn.Net() as net:
        net.opt.use_packing_layout = False
        #  net.opt.lightmode = False
        net.load_param(param)
        net.load_model(model)

        with net.create_extractor() as ex:
            ex.input("in0", ncnn.Mat(x.squeeze(0).numpy()).clone())

            ret, ncnn_out0 = ex.extract("out0")
            ncnn_y = torch.from_numpy(ncnn_out0.numpy()).clone()
            print(y.shape, ncnn_y.shape)
            assert torch.allclose(y, ncnn_y, atol=1e-3), (y - ncnn_y).abs().max()


if __name__ == "__main__":
    torch.manual_seed(20230115)
    main()
