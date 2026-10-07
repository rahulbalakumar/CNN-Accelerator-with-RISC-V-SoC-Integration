"""
send_image.py — Host-side script for the CNN Accelerator RISC-V SoC

Protocol (boot):
  1. Host sends 'S' (start token)
  2. Host sends 9 signed bytes (3x3 kernel weights)
  3. Host sends 2 bytes (bias)
  4. Host sends 1 byte (shift_s)
  5. Host sends 1690 bytes (dense weights)
  6. Host sends 20 bytes (10 dense biases)
  7. Host sends 784 bytes (image)
  8. FPGA sends back 'R' + 1 result byte

Protocol (per image after boot):
  1. Host sends 'S' (start token)
  2. Host sends 784 bytes (image)
  3. FPGA sends back 'R' + 1 result byte
"""

import serial
import time
import sys
import struct
import math
import random

COM_PORT  = 'COM3'
BAUD_RATE = 115200

IMAGE_W   = 28
IMAGE_H   = 28

def send_image(image_bytes, kernel, bias, shift_s, dense_weights, dense_biases, port=COM_PORT, first_run=True):
    assert len(image_bytes) == IMAGE_W * IMAGE_H, "Image must be 784 bytes"
    assert len(kernel) == 9,                      "Kernel must be 9 values"
    assert len(dense_weights) == 1690,            "Dense weights must be 1690 values"
    assert len(dense_biases) == 10,               "Dense biases must be 10 values"
    assert -128 <= bias <= 127 or -32768 <= bias <= 32767, "Bias out of range"
    assert 0 <= shift_s <= 31,                    "shift_s must be 0-31"

    print(f"Connecting to FPGA on {port} at {BAUD_RATE} baud...")
    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=5)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    time.sleep(0.5)

    print("Sending start token...")
    ser.write(b'S')

    if first_run:
        print("Sending kernel weights (9 bytes)...")
        ser.write(bytes([w & 0xFF for w in kernel]))

        print("Sending bias (2 bytes, little-endian)...")
        ser.write(struct.pack('<h', bias))

        print("Sending shift_s (1 byte)...")
        ser.write(bytes([shift_s & 0x1F]))

        print(f"Sending {len(dense_weights)} dense weights...")
        ser.write(bytes([int(w) & 0xFF for w in dense_weights]))

        print("Sending 10 dense biases...")
        for b in dense_biases:
            ser.write(struct.pack('<h', int(b)))

    print(f"Sending {IMAGE_W}x{IMAGE_H} image ({len(image_bytes)} bytes)...")
    ser.write(image_bytes)

    print(f"Waiting for hardware classification result...")
    result_class = None
    while True:
        token = ser.read(1)
        if token == b'R':
            val = ser.read(1)
            if val:
                result_class = int.from_bytes(val, byteorder='little', signed=False)
                break

    ser.close()
    return result_class


def make_dummy_image():
    return bytes([128] * (IMAGE_W * IMAGE_H))


def make_edge_kernel():
    kernel = [-1, -1, -1,
              -1,  8, -1,
              -1, -1, -1]
    bias     = 0
    shift_s  = 0
    return kernel, bias, shift_s


def generate_dummy_dense_weights():
    """Generates random weights for the dense layer to simulate a trained model."""
    weights = [int(random.uniform(-10, 10)) for _ in range(1690)]
    biases = [int(random.uniform(-50, 50)) for _ in range(10)]
    return weights, biases


if __name__ == "__main__":
    image_bytes        = make_dummy_image()
    kernel, bias, shift_s = make_edge_kernel()
    dense_weights, dense_biases = generate_dummy_dense_weights()

    print(f"Kernel: {kernel}  bias={bias}  shift_s={shift_s}")

    print("\n--- Running Full Hardware CNN Accelerator ---")
    predicted_digit = send_image(image_bytes, kernel, bias, shift_s, dense_weights, dense_biases, first_run=True)

    print(f"\n>>> FPGA HARDWARE PREDICTED DIGIT: {predicted_digit} <<<")
    print("Classification was completely performed in hardware!")
