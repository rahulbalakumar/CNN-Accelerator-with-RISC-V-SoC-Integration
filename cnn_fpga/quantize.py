"""
quantize.py - int8 quantisation, bit-exact integer model of the RTL, and test-vector export.

Run from the cnn_fpga folder after train.py:   python quantize.py [--n-vectors 10]

Integer model (matches the RTL bit for bit):
  conv   : acc   = sum_i p_i * w_i + b             p_i uint8 (0..255), w_i int8, b int16
                                                    (mac_unit, adder_tree: 20-bit signed)
  requant: q     = clamp(acc >>> s, -128, 127)      (quant_sat_unit: arithmetic shift = floor)
  relu   : a     = max(q, 0)                        (relu_activation)
  pool   : 2x2 max, stride 2, 26x26 -> 13x13        (max_pool_2d)
  dense  : z_k   = sum_j a_j * W_kj + c_k           a_j 0..127, W int8, c int16, 32-bit acc
                                                    j = 13*row + col (row-major)
  argmax : first index with the largest z_k         (argmax: strict '>' keeps the first)

Scales (symmetric, per tensor):
  pixel   x = p / 255                    (train.py uses the same scaling)
  conv    w = w_q * dw,  dw = max|w| / 127
          acc is in units of dw/255, so b_q = round(b * 255 / dw)
          + 2^(s-1) rounding offset so the floor-shift rounds to nearest
  act     a_real = a * da,  da = 2^s * dw / 255
  dense   W = W_q * dW,  dW = max|W| / 127,   c_q = round(c / (da * dW))
The shift s is chosen by sweeping all legal values and keeping the most accurate
on 10,000 training images (never on the test set).

Outputs:
  weights/quant_params.json             all integer parameters + scales + accuracy
  weights/weights_uart.hex              1722 bytes in UART order (one byte per line)
  ../tb/system/vectors/weights_uart.hex same file, for the SoC testbench
  ../tb/system/vectors/images.hex       N x 784 pixel bytes
  ../tb/system/vectors/expected.hex     N predicted digits (integer model)
  ../tb/system/vectors/labels.hex       N true MNIST labels
  ../tb/system/vectors/logits.hex       N x 10 expected 32-bit logits (two's complement)
"""

import argparse
import json
import os

import numpy as np
import torch

from fpga_model import FPGACNN, load_mnist

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "models", "fpga_cnn.pth")
WEIGHTS_DIR = os.path.join(HERE, "weights")
VEC_DIR = os.path.normpath(os.path.join(HERE, "..", "tb", "system", "vectors"))

INT16_MIN, INT16_MAX = -32768, 32767


# --------------------------------------------------------------------------- integer model
def int_conv_relu(images, wq, bq, s):
    """images uint8 [N,28,28] -> activations int64 [N,26,26] in 0..127."""
    x = images.astype(np.int64)
    acc = np.zeros((x.shape[0], 26, 26), dtype=np.int64)
    for r in range(3):
        for c in range(3):
            acc += x[:, r:r + 26, c:c + 26] * int(wq[3 * r + c])
    acc += int(bq)
    assert acc.min() >= -(1 << 19) and acc.max() < (1 << 19), "20-bit adder_tree overflow"
    q = np.clip(acc >> s, -128, 127)          # numpy >> on signed ints is arithmetic (floor)
    return np.maximum(q, 0)


def int_pool(a):
    n = a.shape[0]
    return a.reshape(n, 13, 2, 13, 2).max(axis=(2, 4)).reshape(n, 169)


def int_dense(feat, Wq, cq):
    z = feat @ Wq.astype(np.int64).T + cq.astype(np.int64)
    assert np.abs(z).max() < (1 << 31), "32-bit dense accumulator overflow"
    return z


def int_forward(images, P):
    feat = int_pool(int_conv_relu(images, P["conv_w"], P["conv_b"], P["shift"]))
    logits = int_dense(feat, P["dense_w"], P["dense_b"])
    return logits, logits.argmax(axis=1)        # np.argmax keeps the first maximum, like the RTL


# --------------------------------------------------------------------------- quantisation
def quantise(sd, shift):
    w = sd["conv.weight"].reshape(9).astype(np.float64)
    b = float(sd["conv.bias"].reshape(()))
    W = sd["fc.weight"].astype(np.float64)
    c = sd["fc.bias"].astype(np.float64)

    dw = np.abs(w).max() / 127.0
    wq = np.clip(np.round(w / dw), -128, 127).astype(np.int64)
    bq = int(np.round(b * 255.0 / dw)) + ((1 << (shift - 1)) if shift > 0 else 0)

    da = (2 ** shift) * dw / 255.0
    dW = np.abs(W).max() / 127.0
    Wq = np.clip(np.round(W / dW), -128, 127).astype(np.int64)
    cq = np.round(c / (da * dW)).astype(np.int64)

    if not (INT16_MIN <= bq <= INT16_MAX) or cq.min() < INT16_MIN or cq.max() > INT16_MAX:
        return None                               # this shift cannot be represented
    return {"conv_w": wq, "conv_b": bq, "shift": shift, "dense_w": Wq, "dense_b": cq,
            "scales": {"dw": dw, "da": da, "dW": dW}}


def accuracy(images, labels, P):
    return float((int_forward(images, P)[1] == labels).mean() * 100.0)


def uart_bytes(P):
    """The exact byte stream send_image.py / the testbench sends after the 'S' token."""
    out = [int(v) & 0xFF for v in P["conv_w"]]
    out += [P["conv_b"] & 0xFF, (P["conv_b"] >> 8) & 0xFF]
    out += [P["shift"] & 0x1F]
    out += [int(v) & 0xFF for v in P["dense_w"].reshape(-1)]     # digit 0's 169 weights first
    for v in P["dense_b"]:
        out += [int(v) & 0xFF, (int(v) >> 8) & 0xFF]
    assert len(out) == 9 + 2 + 1 + 1690 + 20 == 1722
    return out


def write_hex(path, values, width=2):
    mask = (1 << (4 * width)) - 1
    with open(path, "w") as f:
        for v in values:
            f.write(f"{int(v) & mask:0{width}x}\n")


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-vectors", type=int, default=10,
                    help="test images exported for the SoC testbench (default 10: one of each digit)")
    args = ap.parse_args()

    model = FPGACNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    sd = {k: v.detach().numpy() for k, v in model.state_dict().items()}

    train_x, train_y = load_mnist("train")
    test_x, test_y = load_mnist("test")
    calib_x, calib_y = train_x[:10000], train_y[:10000]

    with torch.no_grad():
        fp32_pred = model(torch.from_numpy(test_x).float().unsqueeze(1) / 255.0).argmax(1).numpy()
    fp32_acc = float((fp32_pred == test_y).mean() * 100.0)

    print("shift | calib acc (%)")
    best = None
    for s in range(0, 32):
        P = quantise(sd, s)
        if P is None:
            continue
        acc = accuracy(calib_x, calib_y, P)
        print(f"{s:>5} | {acc:6.2f}")
        if best is None or acc > best[0]:
            best = (acc, P)
    P = best[1]

    int_logits, int_pred = int_forward(test_x, P)
    int8_acc = float((int_pred == test_y).mean() * 100.0)
    agree = float((int_pred == fp32_pred).mean() * 100.0)

    print(f"\nChosen shift s = {P['shift']}")
    print(f"conv  w_q = {P['conv_w'].tolist()}   b_q = {P['conv_b']}")
    print(f"dense W_q range [{P['dense_w'].min()}, {P['dense_w'].max()}], "
          f"c_q = {P['dense_b'].tolist()}")
    print(f"\nTest accuracy  FP32: {fp32_acc:.2f}%   int8 (bit-exact model): {int8_acc:.2f}%")
    print(f"int8 and FP32 predict the same digit on {agree:.2f}% of test images")

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    os.makedirs(VEC_DIR, exist_ok=True)
    stream = uart_bytes(P)
    write_hex(os.path.join(WEIGHTS_DIR, "weights_uart.hex"), stream)
    write_hex(os.path.join(VEC_DIR, "weights_uart.hex"), stream)

    # Test vectors: by default the first test image of each digit 0..9, so every class is exercised.
    if args.n_vectors == 10:
        idx = [int(np.where(test_y == d)[0][0]) for d in range(10)]
    else:
        idx = list(range(args.n_vectors))
    write_hex(os.path.join(VEC_DIR, "images.hex"), test_x[idx].reshape(-1))
    write_hex(os.path.join(VEC_DIR, "expected.hex"), int_pred[idx])
    write_hex(os.path.join(VEC_DIR, "labels.hex"), test_y[idx])
    write_hex(os.path.join(VEC_DIR, "logits.hex"), int_logits[idx].reshape(-1), width=8)

    params = {
        "conv_w": P["conv_w"].tolist(), "conv_b": P["conv_b"], "shift": P["shift"],
        "dense_w": P["dense_w"].tolist(), "dense_b": P["dense_b"].tolist(),
        "scales": P["scales"],
        "accuracy": {"fp32_test": fp32_acc, "int8_test": int8_acc, "int8_fp32_agreement": agree},
        "vectors": {"test_indices": idx, "labels": test_y[idx].tolist(),
                    "expected": int_pred[idx].tolist()},
    }
    with open(os.path.join(WEIGHTS_DIR, "quant_params.json"), "w") as f:
        json.dump(params, f, indent=1)

    print(f"\nVectors for test images {idx}")
    print(f"  labels   {test_y[idx].tolist()}")
    print(f"  expected {int_pred[idx].tolist()}")
    print(f"Wrote {WEIGHTS_DIR}/quant_params.json, weights_uart.hex and {VEC_DIR}/*.hex")


if __name__ == "__main__":
    main()
