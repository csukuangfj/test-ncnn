#!/usr/bin/env python3


import ncnn
import torch

from generate import Foo
from scaling_converter import convert_scaled_to_non_scaled
from custom_layer import RegisterCustomLayer


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

    param = "m.ncnn.param"
    model = "m.ncnn.bin"
    with ncnn.Net() as net:
        RegisterCustomLayer(net)
        #  net.opt.use_packing_layout = False
        #  net.opt.use_fp16_arithmetic = False
        #  net.opt.use_fp16_storage = False
        #  net.opt.use_packing_layout = False
        #  net.opt.lightmode = False
        #  net.opt.lightmode = False
        net.opt.num_threads = 4
        net.load_param(param)
        net.load_model(model)

        states = f.encoder.get_init_state()
        num_encoders = 5

        for kk in range(10):
            print("----iter----", kk)

            x = torch.rand(1, T, num_features)
            cached_len = states[num_encoders * 0 : num_encoders * 1]  # (num_layers, 1)
            cached_avg = states[
                num_encoders * 1 : num_encoders * 2
            ]  # (num_layers, 1, d_model)

            # (num_layers, left_context_len, 1, attention_dim)
            cached_key = states[num_encoders * 2 : num_encoders * 3]

            # (num_layers, left_context_len, 1, attention_dim//2)
            cached_val = states[num_encoders * 3 : num_encoders * 4]

            cached_val2 = states[num_encoders * 4 : num_encoders * 5]

            # (num_layers, 1, d_model, cnn_module_kernel-1)
            cached_conv1 = states[num_encoders * 5 : num_encoders * 6]

            # (num_layers, 1, d_model, cnn_module_kernel-1)
            cached_conv2 = states[num_encoders * 6 : num_encoders * 7]

            y, next_states = m(x, states)

            with net.create_extractor() as ex:
                ex.input("in0", ncnn.Mat(x.squeeze(0).numpy()).clone())

                ex.input(f"in1", ncnn.Mat(cached_len[0].squeeze().numpy()).clone())
                ex.input(f"in2", ncnn.Mat(cached_len[1].squeeze().numpy()).clone())
                ex.input(f"in3", ncnn.Mat(cached_len[2].squeeze().numpy()).clone())
                ex.input(f"in4", ncnn.Mat(cached_len[3].squeeze().numpy()).clone())
                ex.input(f"in5", ncnn.Mat(cached_len[4].squeeze().numpy()).clone())

                ex.input(f"in6", ncnn.Mat(cached_avg[0].squeeze().numpy()).clone())
                ex.input(f"in7", ncnn.Mat(cached_avg[1].squeeze().numpy()).clone())
                ex.input(f"in8", ncnn.Mat(cached_avg[2].squeeze().numpy()).clone())
                ex.input(f"in9", ncnn.Mat(cached_avg[3].squeeze().numpy()).clone())
                ex.input(f"in10", ncnn.Mat(cached_avg[4].squeeze().numpy()).clone())
                #
                ex.input(f"in11", ncnn.Mat(cached_key[0].squeeze().numpy()).clone())
                ex.input(f"in12", ncnn.Mat(cached_key[1].squeeze().numpy()).clone())
                ex.input(f"in13", ncnn.Mat(cached_key[2].squeeze().numpy()).clone())
                ex.input(f"in14", ncnn.Mat(cached_key[3].squeeze().numpy()).clone())
                ex.input(f"in15", ncnn.Mat(cached_key[4].squeeze().numpy()).clone())
                #
                ex.input(f"in16", ncnn.Mat(cached_val[0].squeeze().numpy()).clone())
                ex.input(f"in17", ncnn.Mat(cached_val[1].squeeze().numpy()).clone())
                ex.input(f"in18", ncnn.Mat(cached_val[2].squeeze().numpy()).clone())
                ex.input(f"in19", ncnn.Mat(cached_val[3].squeeze().numpy()).clone())
                ex.input(f"in20", ncnn.Mat(cached_val[4].squeeze().numpy()).clone())
                #
                ex.input(f"in21", ncnn.Mat(cached_val2[0].squeeze().numpy()).clone())
                ex.input(f"in22", ncnn.Mat(cached_val2[1].squeeze().numpy()).clone())
                ex.input(f"in23", ncnn.Mat(cached_val2[2].squeeze().numpy()).clone())
                ex.input(f"in24", ncnn.Mat(cached_val2[3].squeeze().numpy()).clone())
                ex.input(f"in25", ncnn.Mat(cached_val2[4].squeeze().numpy()).clone())
                #
                ex.input(f"in26", ncnn.Mat(cached_conv1[0].squeeze().numpy()).clone())
                ex.input(f"in27", ncnn.Mat(cached_conv1[1].squeeze().numpy()).clone())
                ex.input(f"in28", ncnn.Mat(cached_conv1[2].squeeze().numpy()).clone())
                ex.input(f"in29", ncnn.Mat(cached_conv1[3].squeeze().numpy()).clone())
                ex.input(f"in30", ncnn.Mat(cached_conv1[4].squeeze().numpy()).clone())
                #
                ex.input(f"in31", ncnn.Mat(cached_conv2[0].squeeze().numpy()).clone())
                ex.input(f"in32", ncnn.Mat(cached_conv2[1].squeeze().numpy()).clone())
                ex.input(f"in33", ncnn.Mat(cached_conv2[2].squeeze().numpy()).clone())
                ex.input(f"in34", ncnn.Mat(cached_conv2[3].squeeze().numpy()).clone())
                ex.input(f"in35", ncnn.Mat(cached_conv2[4].squeeze().numpy()).clone())

                ret, ncnn_out0 = ex.extract("out0")
                assert ret == 0, ret

                ncnn_y = torch.from_numpy(ncnn_out0.numpy()).clone()

                y = y.squeeze()
                print("shape", y.shape, ncnn_y.shape)
                print("y", y.reshape(-1)[:10], y.abs().sum())
                print("ncnn_y", ncnn_y.reshape(-1)[:10], ncnn_y.abs().sum())
                print("y.shape", y.shape, ncnn_y.shape)
                assert torch.allclose(y, ncnn_y, atol=1e-2), (y - ncnn_y).abs().max()

                return

                next_cached_len = []
                for i in range(num_encoders):
                    ret, ncnn_out = ex.extract(f"out{i+1}")
                    assert ret == 0, ret
                    ncnn_out = torch.from_numpy(ncnn_out.numpy()).clone()
                    next_cached_len.append(ncnn_out)
                for (torch_s, ncnn_s) in zip(
                    next_states[:num_encoders], next_cached_len
                ):
                    print("cached_len", torch_s.shape, ncnn_s.shape)
                    assert torch.all(torch.eq(torch_s, ncnn_s)), (
                        (torch_s - ncnn_s).abs().max()
                    )

                next_cached_avg = []
                for i in range(num_encoders):
                    ret, ncnn_out = ex.extract(f"out{i+6}")
                    assert ret == 0, ret
                    ncnn_out = torch.from_numpy(ncnn_out.numpy()).clone()
                    next_cached_avg.append(ncnn_out)
                for i, (torch_s, ncnn_s) in enumerate(
                    zip(
                        next_states[num_encoders * 1 : num_encoders * 2],
                        next_cached_avg,
                    )
                ):
                    print(i, "cached_avg", torch_s.shape, ncnn_s.shape)
                    print(
                        i, "cached_avg abs sum", torch_s.abs().sum(), ncnn_s.abs().sum()
                    )
                    print(
                        i,
                        "cached_avg values",
                        torch_s.reshape(-1)[:5],
                        ncnn_s.reshape(-1)[:5],
                    )
                    assert torch.allclose(torch_s, ncnn_s, atol=1e-2), (
                        (torch_s - ncnn_s).abs().max()
                    )

                next_cached_key = []
                for i in range(num_encoders):
                    ret, ncnn_out = ex.extract(f"out{i+11}")
                    assert ret == 0, ret
                    ncnn_out = torch.from_numpy(ncnn_out.numpy()).clone()
                    next_cached_key.append(ncnn_out)
                for i, (torch_s, ncnn_s) in enumerate(
                    zip(
                        next_states[num_encoders * 2 : num_encoders * 3],
                        next_cached_key,
                    )
                ):
                    torch_s = torch_s.squeeze()
                    print(i, "cached_key", torch_s.shape, ncnn_s.shape)
                    print(
                        i, "cached_key abs sum", torch_s.abs().sum(), ncnn_s.abs().sum()
                    )
                    print(
                        i,
                        "cached_key values",
                        torch_s.reshape(-1)[:5],
                        ncnn_s.reshape(-1)[:5],
                    )
                    assert torch.allclose(torch_s, ncnn_s, atol=1e-2), (
                        (torch_s - ncnn_s).abs().max()
                    )

                next_cached_val = []
                for i in range(num_encoders):
                    ret, ncnn_out = ex.extract(f"out{i+16}")
                    assert ret == 0, ret
                    ncnn_out = torch.from_numpy(ncnn_out.numpy()).clone()
                    next_cached_val.append(ncnn_out)
                for i, (torch_s, ncnn_s) in enumerate(
                    zip(
                        next_states[num_encoders * 3 : num_encoders * 4],
                        next_cached_val,
                    )
                ):
                    torch_s = torch_s.squeeze()
                    print(i, "cached_val", torch_s.shape, ncnn_s.shape)
                    print(
                        i, "cached_val abs sum", torch_s.abs().sum(), ncnn_s.abs().sum()
                    )
                    print(
                        i,
                        "cached_val values",
                        torch_s.reshape(-1)[:5],
                        ncnn_s.reshape(-1)[:5],
                    )
                    assert torch.allclose(torch_s, ncnn_s, atol=1e-2), (
                        (torch_s - ncnn_s).abs().max()
                    )

                next_cached_val2 = []
                for i in range(num_encoders):
                    ret, ncnn_out = ex.extract(f"out{i+21}")
                    assert ret == 0, ret
                    ncnn_out = torch.from_numpy(ncnn_out.numpy()).clone()
                    next_cached_val2.append(ncnn_out)
                for i, (torch_s, ncnn_s) in enumerate(
                    zip(
                        next_states[num_encoders * 4 : num_encoders * 5],
                        next_cached_val2,
                    )
                ):
                    torch_s = torch_s.squeeze()
                    print(i, "cached_val2", torch_s.shape, ncnn_s.shape)
                    print(
                        i,
                        "cached_val2 abs sum",
                        torch_s.abs().sum(),
                        ncnn_s.abs().sum(),
                    )
                    print(
                        i,
                        "cached_val2 values",
                        torch_s.reshape(-1)[:5],
                        ncnn_s.reshape(-1)[:5],
                    )
                    assert torch.allclose(torch_s, ncnn_s, atol=1e-2), (
                        (torch_s - ncnn_s).abs().max()
                    )

                next_cached_conv1 = []
                for i in range(num_encoders):
                    ret, ncnn_out = ex.extract(f"out{i+26}")
                    assert ret == 0, ret
                    ncnn_out = torch.from_numpy(ncnn_out.numpy()).clone()
                    next_cached_conv1.append(ncnn_out)
                for i, (torch_s, ncnn_s) in enumerate(
                    zip(
                        next_states[num_encoders * 5 : num_encoders * 6],
                        next_cached_conv1,
                    )
                ):
                    torch_s = torch_s.squeeze()
                    print(i, "cached_conv1", torch_s.shape, ncnn_s.shape)
                    print(
                        i,
                        "cached_conv1 abs sum",
                        torch_s.abs().sum(),
                        ncnn_s.abs().sum(),
                    )
                    print(
                        i,
                        "cached_conv1 values",
                        torch_s.reshape(-1)[:5],
                        ncnn_s.reshape(-1)[:5],
                    )
                    assert torch.allclose(torch_s, ncnn_s, atol=1e-2), (
                        (torch_s - ncnn_s).abs().max()
                    )

                next_cached_conv2 = []
                for i in range(num_encoders):
                    ret, ncnn_out = ex.extract(f"out{i+31}")
                    assert ret == 0, ret
                    ncnn_out = torch.from_numpy(ncnn_out.numpy()).clone()
                    next_cached_conv2.append(ncnn_out)
                for i, (torch_s, ncnn_s) in enumerate(
                    zip(
                        next_states[num_encoders * 6 : num_encoders * 7],
                        next_cached_conv2,
                    )
                ):
                    torch_s = torch_s.squeeze()
                    print(i, "cached_conv2", torch_s.shape, ncnn_s.shape)
                    print(
                        i,
                        "cached_conv2 abs sum",
                        torch_s.abs().sum(),
                        ncnn_s.abs().sum(),
                    )
                    print(
                        i,
                        "cached_conv2 values",
                        torch_s.reshape(-1)[:5],
                        ncnn_s.reshape(-1)[:5],
                    )
                    assert torch.allclose(torch_s, ncnn_s, atol=1e-2), (
                        (torch_s - ncnn_s).abs().max()
                    )

                states = next_states


if __name__ == "__main__":
    torch.manual_seed(20230115)
    main()
