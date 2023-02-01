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

    y, y_lens = m(x, x_lens, states)

    print((x.shape[1] - 7) // 2)

    print("x", x.shape)  # (1, 39, 80)
    print("y", y.shape)  # (1, 16, 384)
    print("y_lens", y_lens)

    y = y.squeeze(0)

    param = "m.ncnn.param"
    model = "m.ncnn.bin"
    with ncnn.Net() as net:
        #  net.opt.use_packing_layout = False
        #  net.opt.lightmode = False
        #  net.opt.lightmode = False
        net.load_param(param)
        net.load_model(model)

        with net.create_extractor() as ex:
            ex.input("in0", ncnn.Mat(x.squeeze(0).numpy()).clone())

            x_lens = x_lens.float()
            ex.input("in1", ncnn.Mat(x_lens.numpy()).clone())

            ret, ncnn_out0 = ex.extract("out0")
            assert ret == 0, ret

            ret, ncnn_out1 = ex.extract("out1")
            assert ret == 0, ret

            ncnn_y = torch.from_numpy(ncnn_out0.numpy()).clone()
            ncnn_y_lens = torch.from_numpy(ncnn_out1.numpy()).clone().int()

            print("shape", y.shape, ncnn_y.shape)
            assert torch.allclose(y, ncnn_y, atol=1e-2), (y - ncnn_y).abs().max()
            assert torch.eq(y_lens, ncnn_y_lens), (y_lens, ncnn_y_lens)


if __name__ == "__main__":
    torch.manual_seed(20230115)
    main()
