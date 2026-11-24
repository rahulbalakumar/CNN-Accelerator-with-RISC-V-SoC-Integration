"""
send_image.py - Run real MNIST digits through the FPGA and collect the measurements.

Usage (board programmed, USB-serial cable on the DE2-115 RS-232 port):
    pip install pyserial numpy torch
    python send_image.py --port COM3                 # 20 test images (default)
    python send_image.py --port COM3 --count 100     # more images
    python send_image.py --port /dev/ttyUSB0 --start 500 --count 50

Protocol (see sw/main.c):
    'W' + 1722 weight bytes          -> 'K'
    'S' + 784 pixel bytes            -> 'R', digit, acc_cycles (u32 LE), uart_cycles (u32 LE)
    'B'                              -> 'B', digit, cpu_cycles (u32 LE)   CPU-only baseline

Every FPGA answer is compared with the bit-exact Python integer model (quantize.py),
so 'matches Python' must be 100%. Accuracy against the MNIST labels is reported
separately. Results are also saved to sw/board_results.csv.
"""

import argparse
import csv
import json
import os
import struct
import sys
import time

import numpy as np
import serial

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "cnn_fpga"))
from fpga_model import load_mnist            # noqa: E402
from quantize import int_forward, uart_bytes  # noqa: E402

CLK_HZ = 50_000_000
PARAMS = os.path.join(HERE, "..", "cnn_fpga", "weights", "quant_params.json")


def load_params():
    with open(PARAMS) as f:
        p = json.load(f)
    return {"conv_w": np.array(p["conv_w"]), "conv_b": int(p["conv_b"]), "shift": int(p["shift"]),
            "dense_w": np.array(p["dense_w"]), "dense_b": np.array(p["dense_b"])}


def read_exact(ser, n, what):
    data = ser.read(n)
    if len(data) != n:
        sys.exit(f"Timeout waiting for {what} (got {len(data)} of {n} bytes). "
                 "Check the port, the baud rate and that KEY0 is not held down.")
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM3")
    ap.add_argument("--baud", type=int, default=115200, help="must match BAUD_RATE in cnn_soc_top")
    ap.add_argument("--start", type=int, default=0, help="first MNIST test image index")
    ap.add_argument("--count", type=int, default=20)
    args = ap.parse_args()

    P = load_params()
    images, labels = load_mnist("test")
    idx = list(range(args.start, args.start + args.count))
    _, py_pred = int_forward(images[idx], P)

    ser = serial.Serial(args.port, args.baud, timeout=5)
    time.sleep(0.3)
    ser.reset_input_buffer()

    # 1. Weights, once.
    t0 = time.perf_counter()
    ser.write(b"W" + bytes(uart_bytes(P)))
    if read_exact(ser, 1, "weight ack") != b"K":
        sys.exit("Bad weight ack")
    print(f"Weights loaded: 1722 bytes in {time.perf_counter() - t0:.3f} s\n")

    # 2. Images.
    rows = []
    print(f"{'img':>5} {'label':>5} {'python':>6} {'fpga':>4}  {'acc cyc':>8} {'uart cyc':>9} {'host ms':>8}")
    for k, i in enumerate(idx):
        t0 = time.perf_counter()
        ser.write(b"S" + images[i].tobytes())
        reply = read_exact(ser, 10, f"result for image {i}")
        host_ms = (time.perf_counter() - t0) * 1000
        token, digit, acc_cyc, uart_cyc = struct.unpack("<cBII", reply)
        if token != b"R":
            sys.exit(f"Bad reply token {token!r} for image {i}")
        rows.append({"index": i, "label": int(labels[i]), "python": int(py_pred[k]), "fpga": digit,
                     "acc_cycles": acc_cyc, "uart_cycles": uart_cyc, "host_ms": round(host_ms, 2)})
        flag = "" if digit == py_pred[k] else "  <-- MISMATCH vs Python"
        print(f"{i:>5} {labels[i]:>5} {py_pred[k]:>6} {digit:>4}  {acc_cyc:>8} {uart_cyc:>9} {host_ms:>8.1f}{flag}")

    # 3. CPU-only baseline on the last image.
    ser.write(b"B")
    token, cpu_digit, cpu_cyc = struct.unpack("<cBI", read_exact(ser, 6, "CPU baseline"))
    ser.close()

    n = len(rows)
    match_py = sum(r["fpga"] == r["python"] for r in rows)
    match_lbl = sum(r["fpga"] == r["label"] for r in rows)
    acc_cyc = np.mean([r["acc_cycles"] for r in rows])
    uart_cyc = np.mean([r["uart_cycles"] for r in rows])
    host_ms = np.mean([r["host_ms"] for r in rows])

    print("\n================ summary ================")
    print(f"FPGA matches Python integer model : {match_py}/{n}  (must be {n}/{n})")
    print(f"FPGA accuracy vs MNIST labels     : {match_lbl}/{n} = {100.0 * match_lbl / n:.1f}%")
    print(f"Accelerator time per image        : {acc_cyc:.0f} cycles = {acc_cyc / CLK_HZ * 1e6:.1f} us")
    print(f"CPU-only software inference       : {cpu_cyc} cycles = {cpu_cyc / CLK_HZ * 1e3:.2f} ms "
          f"(digit {cpu_digit}, FPGA said {rows[-1]['fpga']})")
    print(f"Speedup (CPU cycles / accel cycles): {cpu_cyc / acc_cyc:.0f}x")
    print(f"Image transfer over UART          : {uart_cyc:.0f} cycles = {uart_cyc / CLK_HZ * 1e3:.1f} ms "
          f"at {args.baud} baud")
    print(f"End-to-end per image (host clock) : {host_ms:.1f} ms")

    out = os.path.join(HERE, "board_results.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nPer-image results saved to {out}")


if __name__ == "__main__":
    main()
