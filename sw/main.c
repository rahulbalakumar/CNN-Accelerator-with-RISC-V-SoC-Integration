/*
 * main.c - PicoRV32 firmware for the CNN accelerator SoC.
 *
 * UART protocol (host = send_image.py or the testbench):
 *
 *   'W' + 1722 weight bytes  -> 'K'                                          (1 byte)
 *        9 conv weights (int8), conv bias (int16 LE), shift (u8),
 *        1690 dense weights (int8, digit 0's 169 first), 10 dense biases (int16 LE)
 *        Send once per session; the weights stay loaded for every image after it.
 *
 *   'S' + 784 pixel bytes   -> 'R', digit, acc_cycles (u32 LE), uart_cycles (u32 LE)   (10 bytes)
 *        acc_cycles  : start pulse -> result ready, accelerator only
 *        uart_cycles : time spent receiving the 784 pixels
 *
 *   'B'  -> 'B', digit, cpu_cycles (u32 LE)                                 (6 bytes)
 *        Runs the same integer CNN in software on the last image received.
 *        This is the CPU-only baseline used to compute the speedup.
 *
 * The CPU keeps its own copy of the image and dense weights in RAM because the
 * image BRAM and dense weight memory are write-only from the bus.
 */
#include <stdint.h>

#define UART_BASE          0x10000000
#define IMG_BRAM_BASE      0x20000000
#define OUT_BRAM_BASE      0x30000000
#define MAC_CTRL_BASE      0x40000000
#define GPIO_BASE          0x50000000
#define CLASS_WEIGHTS_BASE 0x60000000

#define UART_DATA       (*(volatile uint32_t*)(UART_BASE + 0x00))
#define UART_STATUS     (*(volatile uint32_t*)(UART_BASE + 0x04))

#define MAC_CTRL        (*(volatile uint32_t*)(MAC_CTRL_BASE + 0x00))
#define MAC_WEIGHTS_0   (*(volatile uint32_t*)(MAC_CTRL_BASE + 0x04))
#define MAC_WEIGHTS_1   (*(volatile uint32_t*)(MAC_CTRL_BASE + 0x08))
#define MAC_WEIGHTS_2   (*(volatile uint32_t*)(MAC_CTRL_BASE + 0x0C))
#define MAC_BIAS        (*(volatile uint32_t*)(MAC_CTRL_BASE + 0x10))
#define MAC_SHIFT       (*(volatile uint32_t*)(MAC_CTRL_BASE + 0x14))
#define MAC_IMG_BASE    (*(volatile uint32_t*)(MAC_CTRL_BASE + 0x18))
#define MAC_IMG_LEN     (*(volatile uint32_t*)(MAC_CTRL_BASE + 0x1C))

#define OUT_RESULT      (*(volatile uint32_t*)(OUT_BRAM_BASE + 0x00))
#define GPIO_OUT        (*(volatile uint32_t*)(GPIO_BASE + 0x00))

#define MAC_CTRL_RESULT_RDY  (1u << 1)

#define IMG_W        28
#define IMAGE_SIZE   (IMG_W * IMG_W)   /* 784 */
#define CONV_W       (IMG_W - 2)       /* 26: valid convolution */
#define POOL_W       (CONV_W / 2)      /* 13 */
#define N_FEAT       (POOL_W * POOL_W) /* 169 */
#define N_CLASS      10
#define KERNEL_TAPS  9

/* ------------------------------------------------------------------ copies for the CPU baseline */
static uint8_t  image[IMAGE_SIZE];
static int8_t   conv_w[KERNEL_TAPS];
static int16_t  conv_b;
static uint8_t  conv_shift;
static int8_t   dense_w[N_CLASS][N_FEAT];
static int16_t  dense_b[N_CLASS];

/* ------------------------------------------------------------------ helpers */
static void uart_putchar(uint8_t c) {
    while ((UART_STATUS & 0x1) == 0);
    UART_DATA = (uint32_t)c;
}

static uint8_t uart_getchar(void) {
    while ((UART_STATUS & 0x2) == 0);
    return (uint8_t)(UART_DATA & 0xFF);
}

static void uart_put_u32(uint32_t v) {
    uart_putchar((uint8_t)(v));
    uart_putchar((uint8_t)(v >> 8));
    uart_putchar((uint8_t)(v >> 16));
    uart_putchar((uint8_t)(v >> 24));
}

static int16_t uart_get_i16(void) {
    uint8_t lo = uart_getchar();
    uint8_t hi = uart_getchar();
    return (int16_t)((uint16_t)hi << 8 | lo);
}

static inline uint32_t get_cycles(void) {
    uint32_t cycles;
    __asm__ volatile ("rdcycle %0" : "=r" (cycles));
    return cycles;
}

static void display_hex(uint32_t val) { GPIO_OUT = val; }

/* ------------------------------------------------------------------ weight loading */
static void load_weights(void) {
    for (int i = 0; i < KERNEL_TAPS; i++)
        conv_w[i] = (int8_t)uart_getchar();
    conv_b     = uart_get_i16();
    conv_shift = uart_getchar() & 0x1F;

    MAC_WEIGHTS_0 = (uint32_t)(uint8_t)conv_w[0]        | (uint32_t)(uint8_t)conv_w[1] << 8
                  | (uint32_t)(uint8_t)conv_w[2] << 16  | (uint32_t)(uint8_t)conv_w[3] << 24;
    MAC_WEIGHTS_1 = (uint32_t)(uint8_t)conv_w[4]        | (uint32_t)(uint8_t)conv_w[5] << 8
                  | (uint32_t)(uint8_t)conv_w[6] << 16  | (uint32_t)(uint8_t)conv_w[7] << 24;
    MAC_WEIGHTS_2 = (uint32_t)(uint8_t)conv_w[8];
    MAC_BIAS      = (uint32_t)(uint16_t)conv_b;
    MAC_SHIFT     = conv_shift;

    /* Dense weights: word address = digit*256 + pixel. Nested loops instead of i/169 and i%169:
       RV32I has no divide instruction, and a software divide per byte is slow enough to drop
       UART bytes at high baud rates. */
    volatile uint32_t *w_mem = (volatile uint32_t *)CLASS_WEIGHTS_BASE;
    for (int n = 0; n < N_CLASS; n++) {
        for (int p = 0; p < N_FEAT; p++) {
            int8_t w = (int8_t)uart_getchar();
            dense_w[n][p] = w;
            w_mem[(n << 8) + p] = (uint32_t)(uint8_t)w;
        }
    }
    for (int n = 0; n < N_CLASS; n++) {
        dense_b[n] = uart_get_i16();
        w_mem[2560 + n] = (uint32_t)(uint16_t)dense_b[n];
    }
}

/* ------------------------------------------------------------------ CPU-only baseline */
/* Same integer arithmetic as the RTL: see cnn_fpga/quantize.py for the definition. */
static int cpu_inference(void) {
    uint8_t act[CONV_W][CONV_W];
    for (int r = 0; r < CONV_W; r++) {
        for (int c = 0; c < CONV_W; c++) {
            int32_t acc = conv_b;
            const uint8_t *px = &image[r * IMG_W + c];
            acc += (int32_t)px[0]          * conv_w[0];
            acc += (int32_t)px[1]          * conv_w[1];
            acc += (int32_t)px[2]          * conv_w[2];
            acc += (int32_t)px[IMG_W]      * conv_w[3];
            acc += (int32_t)px[IMG_W + 1]  * conv_w[4];
            acc += (int32_t)px[IMG_W + 2]  * conv_w[5];
            acc += (int32_t)px[2*IMG_W]    * conv_w[6];
            acc += (int32_t)px[2*IMG_W+1]  * conv_w[7];
            acc += (int32_t)px[2*IMG_W+2]  * conv_w[8];
            acc >>= conv_shift;                       /* arithmetic shift, like >>> */
            if (acc > 127) acc = 127;
            if (acc < 0)   acc = 0;                   /* clamp at -128 then ReLU == clamp at 0 */
            act[r][c] = (uint8_t)acc;
        }
    }

    uint8_t feat[N_FEAT];
    for (int r = 0; r < POOL_W; r++) {
        for (int c = 0; c < POOL_W; c++) {
            uint8_t m = act[2*r][2*c];
            if (act[2*r][2*c+1]   > m) m = act[2*r][2*c+1];
            if (act[2*r+1][2*c]   > m) m = act[2*r+1][2*c];
            if (act[2*r+1][2*c+1] > m) m = act[2*r+1][2*c+1];
            feat[r * POOL_W + c] = m;
        }
    }

    int best = 0;
    int32_t best_val = 0;
    for (int n = 0; n < N_CLASS; n++) {
        int32_t z = dense_b[n];
        for (int j = 0; j < N_FEAT; j++)
            z += (int32_t)feat[j] * dense_w[n][j];
        if (n == 0 || z > best_val) {               /* strict '>' keeps the first max, like argmax.sv */
            best_val = z;
            best = n;
        }
    }
    return best;
}

/* ------------------------------------------------------------------ main loop */
int main(void) {
    display_hex(0x00000000);

    while (1) {
        uint8_t token = uart_getchar();

        if (token == 'B') {
            uint32_t t0 = get_cycles();
            int digit = cpu_inference();
            uint32_t t1 = get_cycles();
            uart_putchar('B');
            uart_putchar((uint8_t)digit);
            uart_put_u32(t1 - t0);
            display_hex(0xB0000000u | (uint32_t)digit);
            continue;
        }
        if (token == 'W') {
            display_hex(0x11111111);
            load_weights();
            uart_putchar('K');
            display_hex(0x22222222);
            continue;
        }
        if (token != 'S') continue;
        display_hex(0x33333333);

        /* Receive the image straight into the shared image BRAM, 4 pixels per word. */
        volatile uint32_t *img_words = (volatile uint32_t *)IMG_BRAM_BASE;
        uint32_t t_uart0 = get_cycles();
        for (int i = 0; i < IMAGE_SIZE; i += 4) {
            uint8_t p0 = uart_getchar();
            uint8_t p1 = uart_getchar();
            uint8_t p2 = uart_getchar();
            uint8_t p3 = uart_getchar();
            img_words[i >> 2] = (uint32_t)p0 | (uint32_t)p1 << 8 | (uint32_t)p2 << 16 | (uint32_t)p3 << 24;
            image[i] = p0; image[i + 1] = p1; image[i + 2] = p2; image[i + 3] = p3;
        }
        uint32_t t_uart1 = get_cycles();
        display_hex(0x44444444);

        MAC_IMG_BASE = 0;
        MAC_IMG_LEN  = IMAGE_SIZE;

        uint32_t t_acc0 = get_cycles();
        MAC_CTRL = 1;                                     /* start pulse */
        while ((MAC_CTRL & MAC_CTRL_RESULT_RDY) == 0);
        uint32_t t_acc1 = get_cycles();
        uint32_t digit = OUT_RESULT & 0xF;

        uart_putchar('R');
        uart_putchar((uint8_t)digit);
        uart_put_u32(t_acc1 - t_acc0);
        uart_put_u32(t_uart1 - t_uart0);
        display_hex(digit);
    }
    return 0;
}
