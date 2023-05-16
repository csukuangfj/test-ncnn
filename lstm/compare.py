#!/usr/bin/env python3

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

    y = m(x, x_lens, h0, c0)
    assert y.shape == (N, 1, encoder_dim)

    param = "m.ncnn.param"
    model = "m.ncnn.bin"
    with ncnn.Net() as net:
        #  net.opt.use_packing_layout = False
        net.load_param(param)
        net.load_model(model)

        with net.create_extractor() as ex:
            ex.input("in0", ncnn.Mat(x.squeeze(0).numpy()).clone())
            ret, ncnn_out0 = ex.extract("out0")

            ncnn_y = torch.from_numpy(ncnn_out0.numpy()).clone()
            y = y.reshape(-1)
            print(y[:10], ncnn_y[:10], y.sum(), ncnn_y.sum(), (y - ncnn_y).abs().max())
            assert torch.allclose(y, ncnn_y, atol=1e-3), (y - ncnn_y).abs().max()


if __name__ == "__main__":
    torch.manual_seed(20230516)
    main()
