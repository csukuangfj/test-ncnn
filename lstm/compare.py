#!/usr/bin/env python3

import ncnn
import torch


@torch.no_grad()
def main():
    m = torch.jit.load("m.pt")
    num_encoder_layers = 12
    encoder_dim = 512
    rnn_hidden_size = 1024

    N = 1
    T = 9
    C = 80
    x = torch.rand(N, T, C)
    x_lens = torch.tensor([T], dtype=torch.int64)
    states = m.model.get_init_states()
    h0, c0 = states[0], states[1]
    assert h0.shape == (num_encoder_layers, N, encoder_dim)
    assert c0.shape == (num_encoder_layers, N, rnn_hidden_size)

    y, y_lens, h1, c1 = m(x, x_lens, h0, c0)
    print(y.shape)
    print(y_lens)
    print(h1.shape)
    print(c1.shape)
    assert y.shape == (N, 1, encoder_dim)

    param = "m.ncnn.param"
    model = "m.ncnn.bin"
    with ncnn.Net() as net:
        #  net.opt.use_packing_layout = False
        net.load_param(param)
        net.load_model(model)

        with net.create_extractor() as ex:
            ex.input("in0", ncnn.Mat(x.squeeze(0).numpy()).clone())
            x_lens = x_lens.float()
            ex.input("in1", ncnn.Mat(x_lens.numpy()).clone())
            ex.input("in2", ncnn.Mat(h0.squeeze(1).numpy()).clone())
            ex.input("in3", ncnn.Mat(c0.squeeze(1).numpy()).clone())
            ret, ncnn_out0 = ex.extract("out0")
            assert ret == 0, ret

            ncnn_y = torch.from_numpy(ncnn_out0.numpy()).clone()
            y = y.squeeze(0)
            print("y shape", y.shape, ncnn_y.shape)

            print(y[0, :10], ncnn_y[0, :10])
            print(y[0, -10:], ncnn_y[0, -10:])
            assert torch.allclose(y, ncnn_y, atol=1e-2), (y - ncnn_y).abs().max()
            print("------------------------------")

            ret, ncnn_out1 = ex.extract("out1")
            assert ret == 0, ret

            ncnn_y_lens = torch.from_numpy(ncnn_out1.numpy()).clone()
            print(y_lens, ncnn_y_lens)
            assert y_lens[0] == ncnn_y_lens[0]

            print("------------h------------------")

            ret, ncnn_out2 = ex.extract("out2")
            assert ret == 0, ret

            h1 = h1.squeeze(1)

            ncnn_h1 = torch.from_numpy(ncnn_out2.numpy()).clone()
            print(h1.shape, ncnn_h1.shape)

            print(h1[0, :10], ncnn_h1[0, :10])
            print(h1[-1, :10], ncnn_h1[-1, :10])
            assert torch.allclose(h1, ncnn_h1, atol=1e-2), (h1 - ncnn_h1).abs().max()

            print("------------c------------------")

            ret, ncnn_out3 = ex.extract("out3")
            assert ret == 0, ret

            ncnn_c1 = torch.from_numpy(ncnn_out3.numpy()).clone()

            c1 = c1.squeeze(1)

            print(c1[0, :10], ncnn_c1[0, :10])
            print(c1[-1, :10], ncnn_c1[-1, :10])
            assert torch.allclose(c1, ncnn_c1, atol=1e-2), (c1 - ncnn_c1).abs().max()


if __name__ == "__main__":
    torch.manual_seed(20230516)
    main()
