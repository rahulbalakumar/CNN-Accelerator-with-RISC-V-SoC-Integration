"""
fpga_model.py - Shared definitions for train.py, extract_weights.py and quantize.py.

The network mirrors the hardware exactly:

    Input   : 28 x 28 x 1 greyscale, pixel/255 (no mean/std normalisation)
    Conv2D  : 1 -> 1 channel, 3x3 kernel, padding=0 (valid)   => 26 x 26
    ReLU
    MaxPool : 2x2, stride 2                                    => 13 x 13
    Flatten : row-major                                        => 169
    Linear  : 169 -> 10
    Argmax

padding=0 because slidingWindowAXIBRAM is instantiated with PADDING=0 in
cnn_soc_top.sv, and dense_layer.sv stores exactly 10 x 169 weights.
"""

import gzip
import os

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
MNIST_RAW = os.path.join(HERE, "data", "MNIST", "raw")

IMG = 28
CONV_OUT = IMG - 2          # 26 (valid convolution)
POOL_OUT = CONV_OUT // 2    # 13
N_FEATURES = POOL_OUT * POOL_OUT  # 169


class FPGACNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0, bias=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(N_FEATURES, 10, bias=True)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)   # row-major flatten: index j = 13*r + c
        return self.fc(x)


def _open(path):
    if os.path.exists(path):
        return open(path, "rb")
    return gzip.open(path + ".gz", "rb")


def load_mnist(split="test"):
    """Return (images uint8 [N,28,28], labels uint8 [N]) from the raw IDX files."""
    prefix = "t10k" if split == "test" else "train"
    with _open(os.path.join(MNIST_RAW, f"{prefix}-images-idx3-ubyte")) as f:
        images = np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape(-1, IMG, IMG)
    with _open(os.path.join(MNIST_RAW, f"{prefix}-labels-idx1-ubyte")) as f:
        labels = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
    return images.copy(), labels.copy()
