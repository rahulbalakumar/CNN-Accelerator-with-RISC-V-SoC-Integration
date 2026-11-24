#!/usr/bin/env bash
# Full-SoC MNIST test with Verilator 5 (free). Run from anywhere:  tb/system/run_verilator.sh
set -e
cd "$(dirname "$0")"
verilator --binary --timing -j 0 -Wno-fatal -Wno-lint -Wno-style \
    --top-module tb_soc_mnist -f files.f -o vsim --Mdir obj_dir
./obj_dir/vsim
