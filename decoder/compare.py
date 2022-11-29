#!/usr/bin/env python3

#!/usr/bin/env python3

import ncnn
import torch


@torch.no_grad()
def main():
    m = torch.jit.load("m.pt")
    context_size = 2
    vocab_size = 500

    x = torch.randint(low=0, high=vocab_size, size=(1, context_size))

    y = m(x).squeeze(1)

    print(x.shape)
    print(y.shape)

    param = "m.ncnn.param"
    model = "m.ncnn.bin"
    with ncnn.Net() as net:
        net.opt.use_packing_layout = False
        net.load_param(param)
        net.load_model(model)

        with net.create_extractor() as ex:
            x = x.to(torch.int32)
            print(x)
            ex.input("in0", ncnn.Mat(x.numpy()).clone())

            ret, ncnn_out0 = ex.extract("out0")

            ncnn_y = torch.from_numpy(ncnn_out0.numpy()).clone()
            print("y", y.shape, ncnn_y.shape)
            print(y[0, :10], ncnn_y[0, :10])
            assert torch.allclose(y, ncnn_y, atol=1e-2), (
                (y - ncnn_y).abs().max(),
                y.mean(),
                ncnn_y.mean(),
            )


if __name__ == "__main__":
    #  torch.manual_seed(20221129)
    main()
