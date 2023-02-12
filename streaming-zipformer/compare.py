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
    cached_len = states[num_encoders * 0 : num_encoders * 1]  # (num_layers, 1)
    cached_avg = states[num_encoders * 1 : num_encoders * 2]  # (num_layers, 1, d_model)

    # (num_layers, left_context_len, 1, attention_dim)
    cached_key = states[num_encoders * 2 : num_encoders * 3]
    print("cached_key", cached_key[0].shape)

    # (num_layers, left_context_len, 1, attention_dim//2)
    cached_val = states[num_encoders * 3 : num_encoders * 4]
    print("cached_val", cached_val[0].shape)  # (num_layers, 64, 1, 96)

    cached_val2 = states[num_encoders * 4 : num_encoders * 5]
    print("cached_val2", cached_val2[0].shape)

    # (num_layers, 1, d_model, cnn_module_kernel-1)
    cached_conv1 = states[num_encoders * 5 : num_encoders * 6]
    print("cached_conv1", cached_conv1[0].shape)  # (num_layers, 1, 384, 30)

    # (num_layers, 1, d_model, cnn_module_kernel-1)
    cached_conv2 = states[num_encoders * 6 : num_encoders * 7]
    print("cached_conv2", cached_conv2[0].shape)  # (num_layers, 1, 384, 30)

    print("x", x.shape)  # (1, 39, 80)
    print(x_lens)  # (39,)
    print("cached_len[0].shape", cached_len[0].shape)  # (2, 1)
    print("cached_len[1].shape", cached_len[1].shape)  # (2, 1)
    print("cached_avg[0].shape", cached_avg[0].shape)  # (2, 1, 384)
    print("cached_avg[1].shape", cached_avg[1].shape)  # (2, 1, 384)

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
        #  net.opt.use_fp16_arithmetic = False
        #  net.opt.use_fp16_storage = False
        #  net.opt.use_packing_layout = False
        #  net.opt.lightmode = False
        #  net.opt.lightmode = False
        net.load_param(param)
        net.load_model(model)

        with net.create_extractor() as ex:
            ex.input("in0", ncnn.Mat(x.squeeze(0).numpy()).clone())

            x_lens = x_lens.float()
            ex.input("in1", ncnn.Mat(x_lens.numpy()).clone())

            ex.input(f"in2", ncnn.Mat(cached_len[0].squeeze().numpy()).clone())
            ex.input(f"in3", ncnn.Mat(cached_len[1].squeeze().numpy()).clone())
            ex.input(f"in4", ncnn.Mat(cached_avg[0].squeeze().numpy()).clone())
            ex.input(f"in5", ncnn.Mat(cached_avg[1].squeeze().numpy()).clone())
            ex.input(f"in6", ncnn.Mat(cached_key[0].squeeze().numpy()).clone())
            ex.input(f"in7", ncnn.Mat(cached_key[1].squeeze().numpy()).clone())
            ex.input(f"in8", ncnn.Mat(cached_val[0].squeeze().numpy()).clone())
            ex.input(f"in9", ncnn.Mat(cached_val[1].squeeze().numpy()).clone())
            ex.input(f"in10", ncnn.Mat(cached_val2[0].squeeze().numpy()).clone())
            ex.input(f"in11", ncnn.Mat(cached_val2[1].squeeze().numpy()).clone())

            ex.input(f"in12", ncnn.Mat(cached_conv1[0].squeeze().numpy()).clone())
            ex.input(f"in13", ncnn.Mat(cached_conv1[1].squeeze().numpy()).clone())
            ex.input(f"in14", ncnn.Mat(cached_conv2[0].squeeze().numpy()).clone())
            ex.input(f"in15", ncnn.Mat(cached_conv2[1].squeeze().numpy()).clone())

            #  ex.input(f"in2", ncnn.Mat(cached_len[0].squeeze().numpy()).clone())
            #  ex.input(f"in3", ncnn.Mat(cached_avg[0].squeeze().numpy()).clone())
            #  ex.input(f"in4", ncnn.Mat(cached_key[0].squeeze().numpy()).clone())
            #  ex.input(f"in5", ncnn.Mat(cached_val[0].squeeze().numpy()).clone())
            #  ex.input(f"in6", ncnn.Mat(cached_conv1[0].squeeze().numpy()).clone())

            # fmt: off
            #  for k in range(2):
            #      print('k', k, f"in{2+k}", f"in{4+k}", f"in{6+k}")
            #      print('k', k, f"in{8+k}", f"in{10+k}", f"in{12+k}")
            #      print('k', k, f"in{14+k}")
            #      ex.input(f"in{2+k}", ncnn.Mat(cached_len[k].squeeze().numpy()).clone())
            #      ex.input(f"in{4+k}", ncnn.Mat(cached_avg[k].squeeze().numpy()).clone())
            #      ex.input(f"in{6+k}", ncnn.Mat(cached_key[k].squeeze().numpy()).clone())
            #      ex.input(f"in{8+k}", ncnn.Mat(cached_val[k].squeeze().numpy()).clone())
            #      ex.input(f"in{10+k}", ncnn.Mat(cached_val2[k].squeeze().numpy()).clone())
            #      ex.input(f"in{12+k}", ncnn.Mat(cached_conv1[k].squeeze().numpy()).clone())
            #      ex.input(f"in{14+k}", ncnn.Mat(cached_conv2[k].squeeze().numpy()).clone())
            # fmt: on

            print("start")
            #  ret, ncnn_out61 = ex.extract("31")
            #  assert ret == 0, ret
            #  ncnn_61 = torch.from_numpy(ncnn_out61.numpy()).clone()
            #  print("ncnn_61.shape", ncnn_61.shape)
            #
            #  ret, ncnn_out51 = ex.extract("51")
            #  assert ret == 0, ret
            #  ncnn_51 = torch.from_numpy(ncnn_out51.numpy()).clone()
            #  print("ncnn_51.shape", ncnn_51.shape)

            #  ret, ncnn_out1 = ex.extract("out1")
            #  assert ret == 0, ret

            ret, ncnn_out0 = ex.extract("out0")
            assert ret == 0, ret

            ncnn_y = torch.from_numpy(ncnn_out0.numpy()).clone()

            y = y.squeeze()
            print("shape", y.shape, ncnn_y.shape)
            print("y", y.reshape(-1)[:10], y.abs().sum())
            print("ncnn_y", ncnn_y.reshape(-1)[:10], ncnn_y.abs().sum())
            print("y.shape", y.shape, ncnn_y.shape)
            #  print("y\n", y[:5, :5], "\n", ncnn_y[:5, :5])
            #  print("y\n", y[-5:, -5:], "\n", ncnn_y[-5:, -5:])
            assert torch.allclose(y, ncnn_y, atol=1e-3), (y - ncnn_y).abs().max()
            return
            ncnn_y_lens = torch.from_numpy(ncnn_out1.numpy()).clone().int()
            assert torch.eq(y_lens, ncnn_y_lens), (y_lens, ncnn_y_lens)
            print("lens", y_lens, ncnn_y_lens)


if __name__ == "__main__":
    torch.manual_seed(20230115)
    main()
