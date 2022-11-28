#!/usr/bin/env python3

#!/usr/bin/env python3

import ncnn
import torch


@torch.no_grad()
def main():
    m = torch.jit.load("m.pt")
    state_dict = m.state_dict()
    weight_ih_l0 = state_dict["lstm.weight_ih_l0"]  # (4*hidden_size, input_size)
    hidden_size = weight_ih_l0.shape[0] // 4
    input_size = weight_ih_l0.shape[1]

    weight_hh_l0 = state_dict["lstm.weight_hh_l0"]  # (4*hidden_size, proj_size)
    proj_size = weight_hh_l0.shape[1]

    N = 1
    T = 5
    x = torch.rand(T, N, input_size)
    h0 = torch.rand(1, N, proj_size)
    c0 = torch.rand(1, N, hidden_size)

    y, hx, cx = m(x, h0, c0)
    assert y.shape == (T, N, proj_size)
    assert hx.shape == (1, N, proj_size)
    assert cx.shape == (1, N, hidden_size)

    y = y.squeeze(1)  # remove batch dim
    hx = hx.squeeze(1)
    cx = cx.squeeze(1)

    param = "m.ncnn.param"
    model = "m.ncnn.bin"
    with ncnn.Net() as net:
        net.opt.use_packing_layout = False
        net.load_param(param)
        net.load_model(model)

        with net.create_extractor() as ex:
            ex.input("in0", ncnn.Mat(x.squeeze(1).numpy()).clone())
            ex.input("in1", ncnn.Mat(h0.squeeze(1).numpy()).clone())
            ex.input("in2", ncnn.Mat(c0.squeeze(1).numpy()).clone())

            ret, ncnn_out0 = ex.extract("out0")
            ret, ncnn_out1 = ex.extract("out1")
            ret, ncnn_out2 = ex.extract("out2")

            ncnn_y = torch.from_numpy(ncnn_out0.numpy()).clone()
            print("y", y.shape, ncnn_y.shape)
            assert torch.allclose(y, ncnn_y, atol=1e-3), (y - ncnn_y).abs().max()

            ncnn_hx = torch.from_numpy(ncnn_out1.numpy()).clone()
            print("hx", hx.shape, ncnn_hx.shape)
            assert torch.allclose(hx, ncnn_hx, atol=1e-3), (hx - ncnn_hx).abs().max()

            ncnn_cx = torch.from_numpy(ncnn_out2.numpy()).clone()
            print("cx", cx.shape, ncnn_cx.shape)
            assert torch.allclose(cx, ncnn_cx, atol=1e-3), (cx - ncnn_cx).abs().max()


if __name__ == "__main__":
    torch.manual_seed(20220924)
    main()
