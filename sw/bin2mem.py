"""
bin2mem.py - Turn firmware.bin into the memory images the SoC needs.

    firmware.hex     16384 x 32-bit words (whole 64 KB RAM, little-endian)
    firmware_k.hex   16384 x 8-bit, byte lane k (k = 0..3)  -> $readmemh in simulation
    firmware_k.mif   same, Quartus MIF                       -> ram_init_file in synthesis

Usage: python3 bin2mem.py firmware.bin <quartus_dir> [<sim_dir> ...]
(.mif files are written only to the first folder, the Quartus project folder)
"""
import os
import sys

RAM_BYTES = 65536
WORDS = RAM_BYTES // 4

data = open(sys.argv[1], "rb").read()
if len(data) > RAM_BYTES:
    sys.exit(f"firmware is {len(data)} bytes, RAM is only {RAM_BYTES}")
data += b"\0" * (RAM_BYTES - len(data))

for n, out in enumerate(sys.argv[2:]):
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "firmware.hex"), "w") as f:
        for i in range(0, RAM_BYTES, 4):
            f.write("%08x\n" % int.from_bytes(data[i:i + 4], "little"))
    for k in range(4):
        lane = data[k::4]
        with open(os.path.join(out, f"firmware_{k}.hex"), "w") as f:
            f.write("".join("%02x\n" % b for b in lane))
        if n > 0:
            continue
        with open(os.path.join(out, f"firmware_{k}.mif"), "w") as f:
            f.write(f"DEPTH = {WORDS};\nWIDTH = 8;\nADDRESS_RADIX = HEX;\nDATA_RADIX = HEX;\nCONTENT\nBEGIN\n")
            f.write("".join("%04X : %02X;\n" % (a, b) for a, b in enumerate(lane)))
            f.write("END;\n")
    print(f"wrote firmware.hex, firmware_[0-3].hex{'/.mif' if n == 0 else ''} -> {out}")
