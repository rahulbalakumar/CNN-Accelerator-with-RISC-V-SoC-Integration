# Build the PicoRV32 firmware and install the memory images.
#
#   make            build + copy hex/mif to the repo root (Quartus) and the testbench folders
#   make CROSS=riscv64-unknown-elf-   if your toolchain uses the riscv64 prefix
#
CROSS   ?= riscv32-unknown-elf-
CC      = $(CROSS)gcc
OBJCOPY = $(CROSS)objcopy
SIZE    = $(CROSS)size

CFLAGS  = -mabi=ilp32 -march=rv32i -O2 -ffreestanding -nostdlib -Wall

# Quartus reads firmware_k.mif from the project folder (repo root);
# the testbenches read firmware_k.hex from the folder the simulator runs in.
INSTALL_DIRS = .. ../tb/system

all: firmware.bin
	python3 bin2mem.py firmware.bin $(INSTALL_DIRS)

start.o: start.S
	$(CC) $(CFLAGS) -c start.S -o start.o

main.o: main.c
	$(CC) $(CFLAGS) -c main.c -o main.o

firmware.elf: start.o main.o sections.ld
	$(CC) $(CFLAGS) -Wl,-Bstatic,-T,sections.ld,--strip-debug start.o main.o -lgcc -o firmware.elf
	$(SIZE) firmware.elf

firmware.bin: firmware.elf
	$(OBJCOPY) -O binary firmware.elf firmware.bin

clean:
	rm -f *.o *.elf *.bin

.PHONY: all clean
