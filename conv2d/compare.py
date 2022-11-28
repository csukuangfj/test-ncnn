#!/usr/bin/env python3

#!/usr/bin/env python3

import ncnn
import torch


@torch.no_grad()
def main():
    m = torch.jit.load("m.pt")
    num_features = 80

    T = 20
    x = torch.rand(1, T, num_features)

    y = m(x).squeeze(0)

    print(x.shape)
    print(y.shape)

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
