"""
extract_weights.py - Dump the trained FP32 parameters to ./weights/ for inspection.

Outputs:
    conv_weights_fp32.txt   9 values, row-major (w0 top-left, w8 bottom-right)
    conv_bias_fp32.txt      1 value
    dense_weights_fp32.txt  [10, 169], row i = weights for digit i
    dense_bias_fp32.txt     10 values

Run from the cnn_fpga folder after train.py. quantize.py reads the .pth directly,
so these text files are for the report and for checking by eye.
"""

import os

import numpy as np
import torch

from fpga_model import FPGACNN

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "models", "fpga_cnn.pth")
WEIGHTS_DIR = os.path.join(HERE, "weights")
os.makedirs(WEIGHTS_DIR, exist_ok=True)

model = FPGACNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
sd = {k: v.detach().numpy() for k, v in model.state_dict().items()}

conv_w = sd["conv.weight"].reshape(9)
conv_b = sd["conv.bias"].reshape(1)
fc_w = sd["fc.weight"]          # [10, 169]
fc_b = sd["fc.bias"]            # [10]

np.savetxt(os.path.join(WEIGHTS_DIR, "conv_weights_fp32.txt"), conv_w,
           header="Conv2D kernel weights - 9 values, row-major (w0 top-left, w8 bottom-right)")
np.savetxt(os.path.join(WEIGHTS_DIR, "conv_bias_fp32.txt"), conv_b,
           header="Conv2D bias - 1 value")
np.savetxt(os.path.join(WEIGHTS_DIR, "dense_weights_fp32.txt"), fc_w,
           header="Dense (fc) weights - shape [10, 169]\n"
                  "Row i = weights for digit i (0..9)\n"
                  "Col j = weight connecting pooled pixel j (j = 13*row + col) to digit i")
np.savetxt(os.path.join(WEIGHTS_DIR, "dense_bias_fp32.txt"), fc_b,
           header="Dense (fc) biases - 10 values, one per digit")

print("Conv kernel (row-major):")
print(np.array2string(conv_w.reshape(3, 3), precision=4))
print(f"Conv bias: {conv_b[0]:+.4f}")
print(f"Dense weights: shape {fc_w.shape}, range [{fc_w.min():+.4f}, {fc_w.max():+.4f}]")
print(f"Dense biases : {np.array2string(fc_b, precision=3)}")
print(f"Saved to {WEIGHTS_DIR}")
