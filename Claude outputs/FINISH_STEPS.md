# Finishing the CNN accelerator: steps for each member

Everything in the code is done and verified in simulation. What is left needs Quartus, the DE2-115 board and the report.

## Verified results (simulation)

| Item | Result |
|---|---|
| FP32 test accuracy (10,000 images) | 92.42 % |
| int8 test accuracy (bit-exact integer model) | 92.40 % |
| int8 and FP32 agree on | 99.66 % of test images |
| Full SoC vs Python model (10 digits, one of each) | 100 / 100 logits and 10 / 10 digits identical |
| Accelerator time per image | 830 cycles = 16.6 µs at 50 MHz |
| Same network in software on the PicoRV32 | 3,928,305 cycles = 78.6 ms |
| Compute speedup | 4,733× |
| Sanity check: one dense weight sign flipped | 7 logit errors caught |

Test digit 5 (MNIST test image 8) is predicted as 6 by both Python and the FPGA. That is a model error, not a hardware error. The testbench compares against the Python integer model, not the label.

## What changed and why

| File | Change |
|---|---|
| `cnn_fpga/fpga_model.py` | **New.** The one shared model definition: `padding=0`, 169 → 10. It also loads MNIST from the raw files. |
| `cnn_fpga/train.py` | Uses the shared model. The old one said `padding=1` and 196 inputs, which does not match the hardware. |
| `cnn_fpga/extract_weights.py` | Rewritten: the old file had a syntax error (`comments="` was cut off). |
| `cnn_fpga/quantize.py` | **New.** int8 quantisation, the bit-exact integer model, a shift sweep, accuracy, and export of the weights and test vectors. |
| `cnn_fpga/models/fpga_cnn.pth`, `weights/*` | Retrained model, FP32 text dumps, `quant_params.json`, `weights_uart.hex`. |
| `sw/main.c` | New UART protocol (below). Cycle counts are sent back to the host. The CPU-only baseline is the `'B'` command. The dense weights are loaded without division. |
| `sw/Makefile`, `sw/bin2mem.py` | `make` now also writes the 4 `.mif` files to the repo root (for Quartus) and the `.hex` files to `tb/system`. |
| `sw/send_image.py` | Rewritten for the board: real weights, real digits, checks against Python, prints every measurement, saves a CSV. |
| `tb/system/*` | **New.** The full-SoC test with real data, plus Verilator and ModelSim run scripts and the saved log. |
| `cnn_soc_top.sdc` | **New.** 50 MHz clock and false paths for the buttons, UART and 7-segment pins. |
| `.gitignore` | `*.qsf` and `*.qpf` are no longer ignored, which is why the Quartus project never got committed. |
| Deleted | `sw/cpu_baseline.c` (196-input version, replaced by `'B'`); `tb/soc_integration_jaindu/tb_cnn_soc_top.sv` and its firmware copies (dummy weights, old protocol, replaced by `tb_soc_mnist`). |

UART protocol:

| Host sends | FPGA replies |
|---|---|
| `'W'` + 1722 weight bytes | `'K'` |
| `'S'` + 784 pixels | `'R'`, digit, accelerator cycles (u32), UART receive cycles (u32) |
| `'B'` | `'B'`, digit, CPU software cycles (u32) for the last image |

The 1722 weight bytes are: 9 conv weights, conv bias as 2 bytes, shift as 1 byte, 1690 dense weights (digit 0 first), and 10 dense biases as 2 bytes each.

Weights are sent with their own command, so running the script twice without pressing reset still works.

---

## Rahul: apply, check and push (≈ 30 min, first)

1. Apply the patch on top of the latest `feature/soc-integration`:
   ```
   git checkout feature/soc-integration
   git pull
   git am finish_project.patch
   ```
   If `git am` complains, unzip `finish_project_files.zip` over the repo instead and delete the files listed under "Deleted" above.
2. Run the full-SoC test in ModelSim. In the ModelSim console:
   ```
   cd <repo>/tb/system
   do run_modelsim.do
   ```
   It must end with `PASS: FPGA matches the Python integer model bit for bit`. It is about 60 ms of simulated time, so it takes a few minutes. Screenshot the last 15 lines for the report.
3. Push the branch and tell the other two to pull.
4. After the board run works, merge `feature/soc-integration` into `main`.

## Jaindu: Quartus, timing, power (≈ 2 h)

1. Pull. The new `firmware_0..3.mif` files are already in the repo root, so no RISC-V compiler is needed unless `main.c` changes again.
2. Open your project with `cnn_soc_top` as the top level. The source list must be exactly the RTL files in `tb/system/files.f`; do not add the testbench. Remove the old `slidingWindowAXI.sv` / `lineBufferAXI.sv` from the project if they are in it.
3. Add the constraints: *Assignments → Settings → TimeQuest → add `cnn_soc_top.sdc`*.
4. Full compile.
5. *TimeQuest → Report Fmax Summary* and *Report Setup Summary* for the **Slow 1200mV 85C** model. Record the Fmax and the worst setup slack. Slack must be ≥ 0 at 20 ns.
6. If the slack is negative, send the worst path (*Report Timing*, top path) to Rakesh. The most likely path is the dense-layer weight mux into the multiply-accumulate.
7. *Processing → PowerPlay Power Analyzer*, with a default toggle rate of 12.5 %. Record total, core dynamic and static power.
8. Commit `cnn_soc_top.qpf`, `cnn_soc_top.qsf` and the `.sdc`. Do not commit `db/`, `incremental_db/` or `output_files/`.
9. Record from the fit report: LEs, registers, M9K blocks, embedded multipliers, memory bits.

## Rakesh: model, quantisation, viva (≈ 1.5 h)

1. Pull and run `cd cnn_fpga && python quantize.py`. It must print int8 accuracy 92.40 % and shift 11.
   - Do **not** rerun `train.py` unless the model is changed on purpose. A new model means new vectors, and the simulation must be rerun.
2. Read `quantize.py` until you can explain these on the whiteboard:
   - per-tensor scales $\Delta_w = \max|w|/127$;
   - why the conv bias is $b_q = \mathrm{round}(255\,b/\Delta_w) + 2^{s-1}$ (pixel scale, plus a rounding offset because the hardware shift floors);
   - why the dense bias is divided by $\Delta_a\,\Delta_W$;
   - why argmax is unchanged by a positive scale.
3. Make the accuracy table for the report:

   | Model | Test accuracy |
   |---|---|
   | FP32 | 92.42 % |
   | int8 | 92.40 % |
   | int8 vs FP32 agreement | 99.66 % |

   Add the shift sweep from the `quantize.py` printout.
4. **Only if Jaindu's timing fails** on the dense layer: register the dense-layer product before the accumulate (one extra pipeline stage, `pixel_count` stays the same). Then rerun `tb/system` to prove it is still bit-exact.

## Everyone: board session (≈ 1.5 h)

1. Program the DE2-115 with the `.sof`. Press KEY0: HEX shows `00000000`.
2. Connect the RS-232 port to the PC (USB-serial adapter) and find the COM port.
3. Run:
   ```
   pip install pyserial numpy torch
   python sw/send_image.py --port COM3 --count 20
   python sw/send_image.py --port COM3 --start 0 --count 200
   ```
4. Pass means `FPGA matches Python integer model: N/N`. The label accuracy should be about 92 %.
5. Keep `sw/board_results.csv` and a photo of HEX0 for the report and the demo.

| Symptom | Likely cause |
|---|---|
| `Timeout waiting for weight ack` | Wrong COM port, cable on the wrong connector, or the board not programmed. Press KEY0 and retry. |
| HEX stuck at `11111111` | Weight bytes are being lost; the baud rates do not match. |
| Digits differ from Python | Timing failure (check the slack), or old `.mif` files compiled in (recompile after pulling). |

## Measurements for the report

| Quantity | Value | Source |
|---|---|---|
| Accelerator latency | 830 cycles (16.6 µs) | sim + board |
| CPU-only inference (PicoRV32, RV32I, no hardware multiply) | 3,928,305 cycles (78.6 ms) | sim + board |
| Compute speedup | 4,733× | |
| Image over UART at 115200 baud | ≈ 68 ms (784 × 10 bits) | board (`uart cyc`) |
| End-to-end per image, accelerated | ≈ 68 ms | board |
| End-to-end per image, CPU only | ≈ 68 + 78.6 = 147 ms | |
| End-to-end speedup | ≈ 2.2× | |
| Weights sent once | 1722 bytes ≈ 150 ms | board |
| Pixel reads per image | 784 (each pixel once, line buffers) vs 6,084 for 9 reads per window | design |
| Fmax / slack / LEs / M9K / multipliers / power | — | Jaindu |
| Board hit rate vs Python / vs labels | — / — | board |

The systems-performance story for the viva:
- The accelerator removes compute as the bottleneck: 78.6 ms becomes 16.6 µs.
- Now the UART is 99.97 % of the end-to-end time.
- The next steps would be a faster baud rate (921600 gives about 8.5 ms per image), a DMA from the UART into the image BRAM, or double buffering so the next image arrives while the current one is classified.
