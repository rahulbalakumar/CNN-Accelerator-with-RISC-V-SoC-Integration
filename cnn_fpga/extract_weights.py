"""
extract_weights.py — Load the trained FPGA CNN and extract FP32 parameters.

Outputs (in ./weights/):
    conv_weights_fp32.txt   shape [1,1,3,3] flattened to 9 values  (row-major)
    conv_bias_fp32.txt      1 value
    dense_weights_fp32.txt  shape [10, 196]  — one row per digit class
    dense_bias_fp32.txt     10 values

STOP HERE — do NOT quantize until the FPGA arithmetic has been inspected.
"""

import torch
import torch.nn as nn
import numpy as np
import os

MODEL_PATH  = "./models/fpga_cnn.pth"
WEIGHTS_DIR = "./weights"
os.makedirs(WEIGHTS_DIR, exist_ok=True)


class FPGACNN(nn.Module):
    """
    Software reference model of the FPGA CNN accelerator.
    Must be identical to the class defined in train.py.
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=1, out_channels=1,
            kernel_size=3, stride=1, padding=1, bias=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc   = nn.Linear(196, 10, bias=True)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Trained model not found at '{MODEL_PATH}'.\n"
        "Please run train.py first."
    )

model = FPGACNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()
print(f"[INFO] Loaded model from: {MODEL_PATH}")


conv_weight = model.conv.weight.detach().cpu()
conv_bias   = model.conv.bias.detach().cpu()

print(f"\n[CONV WEIGHTS]  shape: {list(conv_weight.shape)}")
print(f"[CONV BIAS]     shape: {list(conv_bias.shape)}")

conv_weight_flat = conv_weight.reshape(-1).numpy()

print("\nConvolution kernel (row-major):")
for i, v in enumerate(conv_weight_flat):
    row, col = divmod(i, 3)
    print(f"  w[{row}][{col}] = w{i} = {v:+.6f}")

print(f"\nConvolution bias:")
print(f"  b = {conv_bias.item():+.6f}")


dense_weight = model.fc.weight.detach().cpu()
dense_bias   = model.fc.bias.detach().cpu()

print(f"\n[DENSE WEIGHTS] shape: {list(dense_weight.shape)}")
print(f"[DENSE BIAS]    shape: {list(dense_bias.shape)}")

print("\nDense weight stats per neuron (digit class):")
print(f"  {'Digit':>5} | {'min':>10} | {'max':>10} | {'mean':>10} | {'std':>10}")
print("  " + "-" * 52)
for i in range(10):
    w = dense_weight[i].numpy()
    print(f"  {i:>5} | {w.min():>10.5f} | {w.max():>10.5f} | "
          f"{w.mean():>10.5f} | {w.std():>10.5f}")

print("\nDense biases:")
for i, b in enumerate(dense_bias.numpy()):
    print(f"  b[{i}] = {b:+.6f}")


np.savetxt(
    os.path.join(WEIGHTS_DIR, "conv_weights_fp32.txt"),
    conv_weight_flat,
    header="Conv2D kernel weights — 9 values, row-major (w0 top-left, w8 bottom-right)",
    comments="
)

np.savetxt(
    os.path.join(WEIGHTS_DIR, "conv_bias_fp32.txt"),
    conv_bias.numpy(),
    header="Conv2D bias — 1 value",
    comments="
)

np.savetxt(
    os.path.join(WEIGHTS_DIR, "dense_weights_fp32.txt"),
    dense_weight.numpy(),
    header=(
        "Dense (fc) weights — shape [10, 196]\n"
        "
        "
    ),
    comments="
)

np.savetxt(
    os.path.join(WEIGHTS_DIR, "dense_bias_fp32.txt"),
    dense_bias.numpy(),
    header="Dense (fc) biases — 10 values, one per digit class",
    comments="
)

print("\n[INFO] FP32 weight files saved:")
for fname in ["conv_weights_fp32.txt", "conv_bias_fp32.txt",
              "dense_weights_fp32.txt", "dense_bias_fp32.txt"]:
    path = os.path.join(WEIGHTS_DIR, fname)
    print(f"  {path}")


print("\n[INFO] Sanity check — running one dummy image through the model...")
dummy = torch.zeros(1, 1, 28, 28)
with torch.no_grad():
    logits = model(dummy)
    pred   = logits.argmax(dim=1).item()
print(f"  Logits : {[f'{v:.3f}' for v in logits[0].numpy()]}")
print(f"  Argmax : {pred}")


print("\n" + "="*55)
print("  EXTRACTED PARAMETER SUMMARY")
print("="*55)
print(f"  {'Parameter':<25} {'Shape':<15} {'Count':>6}")
print("  " + "-"*45)
print(f"  {'conv.weight':<25} {'[1,1,3,3]':<15} {conv_weight.numel():>6}")
print(f"  {'conv.bias':<25} {'[1]':<15} {conv_bias.numel():>6}")
print(f"  {'fc.weight':<25} {'[10,196]':<15} {dense_weight.numel():>6}")
print(f"  {'fc.bias':<25} {'[10]':<15} {dense_bias.numel():>6}")
print("  " + "-"*45)
total = (conv_weight.numel() + conv_bias.numel() +
         dense_weight.numel() + dense_bias.numel())
print(f"  {'TOTAL':<25} {'':<15} {total:>6}")
print("="*55)

print("""
╔══════════════════════════════════════════════════════╗
║  STOP — DO NOT QUANTIZE YET                          ║
║                                                      ║
║  Next step: inspect the FPGA SystemVerilog to        ║
║  determine:                                          ║
║    • Input pixel bit-width (8-bit unsigned 0-255)    ║
║    • Weight bit-width in datapath_top / dense_layer  ║
║    • Accumulator width                               ║
║    • Shift / scale operation in quant_sat_unit       ║
║    • Signed / unsigned representation                ║
║    • Bias representation                             ║
║                                                      ║
║  Then design quantize.py with a correct scale.       ║
╚══════════════════════════════════════════════════════╝
""")
