# Introduction

This repo contains test code about
converting PyTorch models to ncnn via PNNX.

```bash
git clone https://github.com/csukuangfj/ncnn
git submodule init
git submodule update python/pybind11

python3 setup.py bdist_wheel
pip install ./dist/*.whl

cd tools/pnnx
mkdir build
make -j
./src/pnnx
```
