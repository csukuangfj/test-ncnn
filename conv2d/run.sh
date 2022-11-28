#!/usr/bin/env bash
set -ex

./generate.py
ls -lh m.pt

pnnx ./m.pt

cat m.ncnn.param
# it prints something like below:
# 7767517
# 4 6
# Input                    in0                      0 1 in0
# Input                    in1                      0 1 in1
# Input                    in2                      0 1 in2
# LSTM2                    lstm2_0                  3 3 in0 in1 in2 out0 out1 out2 0=5 1=60 2=0 3=4

# 0=5, number of outputs, i.e., hidden size
# 1=60, hidden_size * input_size * 4 = 5 * 3 * 4 = 60
# 2=0  0 means unidirectional, we only support unidirectional for LSTM with projection
# 3=4, proj_size

./compare.py
