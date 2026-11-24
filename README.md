# CNN Accelerator with RISC-V SoC Integration

An FPGA system-on-chip that classifies handwritten digits (MNIST, 28×28) using a
custom CNN accelerator controlled by a PicoRV32 RISC-V soft core.

**Target:** Terasic DE2-115 (Cyclone IV E, EP4CE115F29C7), 50 MHz

## Architecture

```
Laptop --UART--> PicoRV32 --bus--> Sliding window (3x3) --> Conv datapath --> Max pool --> Dense --> Argmax
                    ^                                                                                  |
                    +----------------------------- result register / status --------------------------+
```

1. The host sends the weights and a 28×28 image over UART.
2. Firmware loads the weights into memory-mapped registers and streams the pixels to the accelerator.
3. The accelerator runs **Conv 3×3 (valid) → ReLU → MaxPool 2×2 → Dense 169→10 → Argmax** entirely in hardware.
4. The firmware reads the predicted digit, shows it on the 7-segment display, and returns it over UART.

**Model:** `Conv2d(1→1, 3×3, padding=0)` → ReLU → MaxPool(2) → Linear(169→10), giving 1,710 parameters (`cnn_fpga/train.py`).

## Memory map

| Address | Device |
|---|---|
| `0x0000_0000` | RAM, 64 KB (firmware) |
| `0x1000_0000` | UART: `+0x0` data, `+0x4` status (bit 0 TX ready, bit 1 RX valid) |
| `0x2000_0000` | Pixel stream into the sliding window (write-only) |
| `0x3000_0000` | Classification result (read clears the ready flag) |
| `0x4000_0000` | Conv control: `+0x0` enable / status, `+0x4..0xC` weights, `+0x10` bias, `+0x14` shift |
| `0x5000_0000` | GPIO → 7-segment displays |
| `0x6000_0000` | Dense weights (`+1024·n + 4·i`) and biases (`+0x2800 + 4·n`) |

## Repository layout

```
rtl/soc_integration_jaindu/      SoC top, PicoRV32, interconnect, UART, GPIO
rtl/dataflow_memory_rahul/       Image BRAM, image reader, line buffers, sliding window
rtl/datapath_arithmetic_rakesh/  MACs, adder tree, quantise/saturate, ReLU, pool, dense, argmax
tb/             Block testbenches
sw/             Firmware (start.S, main.c, linker script, Makefile) and host script
cnn_fpga/       Training, quantisation and bit-exact integer model (PyTorch)
tb/system/      Full-SoC test with real MNIST digits
```

## Team

| Member | Area |
|---|---|
| 1 | Datapath and arithmetic: MAC, adder tree, quantization, ReLU, pooling |
| 2 | Dataflow and memory: line buffers, 3×3 sliding window, padding |
| 3 | SoC integration: CPU, interconnect, peripherals, firmware, timing |

## Status

### Done
- **Dataflow and memory (Member 2):** shared dual-port image BRAM, image reader, BRAM line buffers and
  3×3 sliding window (189.9 MHz standalone), each with its own testbench, plus an end-to-end image-path test.
- **Datapath (Member 1):** 9 parallel unsigned-pixel × int8 MACs, pipelined adder tree, shift/saturate, ReLU,
  2×2 max pool, dense 169→10 with CPU-loadable weights, argmax.
- **SoC (Member 3):** PicoRV32, address decoder, UART, GPIO, firmware, full Quartus fit.
- **Model:** hardware-matched training (`padding=0`, 169 inputs) and int8 quantisation with a bit-exact
  integer model (`cnn_fpga/quantize.py`). Test accuracy: FP32 92.42 %, int8 92.40 %.
- **Full-SoC simulation with real weights and real MNIST digits** (`tb/system/tb_soc_mnist.sv`):
  all 100 logits and 10 digits match the Python integer model bit for bit.
  Accelerator: 830 cycles per image. The same network in software on the PicoRV32 takes 3,928,305 cycles.

### To do

| Owner | Item |
|---|---|
| 3 | Commit the Quartus project (`.qpf`/`.qsf`), add `cnn_soc_top.sdc`, report Fmax and setup slack |
| All | Board run with `sw/send_image.py`: record hit rate, cycle counts and UART time |
| 3 | Power estimate (PowerPlay) |
| All | Report, demo, viva |

## Build and run

**1. Train, quantise and export test vectors** (Python 3, `torch`, `numpy`)
```
cd cnn_fpga
python train.py           # FP32 model -> models/fpga_cnn.pth
python quantize.py        # int8 model, accuracy, weights/quant_params.json, tb/system/vectors/*.hex
```

**2. Build the firmware** (RV32I GCC; copies `firmware_k.mif` to the repo root and `.hex` to `tb/system`)
```
cd sw && make             # or: make CROSS=riscv64-unknown-elf-
```

**3. Full-SoC simulation** (real weights, 10 real digits, bit-exact check)
```
tb/system/run_verilator.sh                    # Verilator 5
cd tb/system  then  do run_modelsim.do        # ModelSim / Questa
```

**4. Board:** compile `cnn_soc_top` in Quartus (with `cnn_soc_top.sdc`), program the DE2-115, then
```
pip install pyserial
python sw/send_image.py --port COM3 --count 20
```

UART protocol: `'W'` + 1722 weight bytes → `'K'`; `'S'` + 784 pixels → `'R'`, digit, accelerator cycles,
UART cycles; `'B'` → CPU-only software inference of the last image with its cycle count.

## Known limitations

- Square images only: the padding logic uses `ROW_LENGTH` as the image height.
- One convolution filter, so accuracy is limited compared with multi-filter CNNs.