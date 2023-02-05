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

    num_encoders = 5
    cached_len = states[0]  # (num_layers, 1)
    cached_avg = states[num_encoders]  # (num_layers, 1, d_model)

    # (num_layers, left_context_len, 1, attention_dim)
    cached_key = states[num_encoders * 2]
    print("cached_key", cached_key.shape)

    cached_val = states[num_encoders * 3]
    print("cached_val", cached_val.shape)

    print("x", x.shape)  # (1, 39, 80)
    print(x_lens)  # (39,)
    print(cached_len.shape)  # (2, 1)
    print(cached_avg.shape)  # (2, 1, 384)

    y, y_lens = m(x, x_lens, states)

    print((x.shape[1] - 7) // 2)

    print("x", x.shape)  # (1, 39, 80)
    print("y", y.shape)  # (1, 16, 384)
    print("y_lens", y_lens)

    """
[1, 39, 80]
after subsampling,
[1, 16, 384]

left_context_len: 64, chunk size 16
2*16 + 64 - 1 = 95, rel positional encoding: pe.shape [1, 95, 384]
    """

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

            ex.input("in2", ncnn.Mat(cached_len.squeeze().numpy()).clone())
            ex.input("in3", ncnn.Mat(cached_avg.squeeze().numpy()).clone())
            ex.input("in4", ncnn.Mat(cached_key.squeeze().numpy()).clone())
            ex.input("in5", ncnn.Mat(cached_val.squeeze().numpy()).clone())

            #  ret, ncnn_out49 = ex.extract("49")
            #  assert ret == 0, ret
            #  ncnn_49 = torch.from_numpy(ncnn_out49.numpy()).clone()
            #  print("ncnn_49.shape", ncnn_49.shape)
            #
            #  ret, ncnn_out51 = ex.extract("51")
            #  assert ret == 0, ret
            #  ncnn_51 = torch.from_numpy(ncnn_out51.numpy()).clone()
            #  print("ncnn_51.shape", ncnn_51.shape)

            ret, ncnn_out1 = ex.extract("out1")
            assert ret == 0, ret

            ret, ncnn_out0 = ex.extract("out0")
            assert ret == 0, ret

            ncnn_y = torch.from_numpy(ncnn_out0.numpy()).clone()
            ncnn_y_lens = torch.from_numpy(ncnn_out1.numpy()).clone().int()

            y = y.squeeze()
            print("y.shape", y.shape, ncnn_y.shape)
            print("y\n", y[:5, :5], "\n", ncnn_y[:5, :5])
            print("y\n", y[-5:, -5:], "\n", ncnn_y[-5:, -5:])
            assert torch.allclose(y, ncnn_y, atol=1e-3), (y - ncnn_y).abs().max()
            assert torch.eq(y_lens, ncnn_y_lens), (y_lens, ncnn_y_lens)
            print("lens", y_lens, ncnn_y_lens)


if __name__ == "__main__":
    torch.manual_seed(20230115)
    main()
