#!/usr/bin/env python3

#!/usr/bin/env python3

import ncnn
import torch


@torch.no_grad()
def main():
    m = torch.jit.load("m.pt")
    encoder_dim = 512
    decoder_dim = 512
    vocab_size = 500

    encoder_out = torch.rand(1, encoder_dim)
    decoder_out = torch.rand(1, decoder_dim)

    y = m(encoder_out, decoder_out)

    print(y.shape)  # (1, 500)

    param = "m.ncnn.param"
    model = "m.ncnn.bin"
    with ncnn.Net() as net:
        net.opt.use_packing_layout = False
        net.load_param(param)
        net.load_model(model)

        with net.create_extractor() as ex:
            ex.input("in0", ncnn.Mat(encoder_out.numpy()).clone())
            ex.input("in1", ncnn.Mat(decoder_out.numpy()).clone())

            ret, ncnn_out0 = ex.extract("out0")  # its shape is (500,)

            ncnn_y = torch.from_numpy(ncnn_out0.numpy()).clone()
            assert ncnn_y.shape == (1, 500)
            assert y.shape == (1, 500)
            print(y[0, :10])
            print(ncnn_y[0, :10])
            assert torch.allclose(y, ncnn_y, atol=1e-2), (
                (y - ncnn_y).abs().max(),
                y.mean(),
                ncnn_y.mean(),
            )


if __name__ == "__main__":
    #  torch.manual_seed(20221129)
    main()
