








static void uart_putchar(char c) {
    while ((UART_STATUS & 0x1) == 0);
    UART_DATA = (uint32_t)c;
}

static char uart_getchar(void) {
    while ((UART_STATUS & 0x2) == 0);
    return (char)(UART_DATA & 0xFF);
}

static void uart_send_byte(uint8_t b) {
    uart_putchar((char)b);
}

static void display_hex(uint32_t val) {
    GPIO_OUT = val;
}

static inline uint32_t get_cycles(void) {
    uint32_t cycles;
    __asm__ volatile ("rdcycle %0" : "=r" (cycles));
    return cycles;
}

static void load_weights(const int8_t weights[KERNEL_TAPS], int16_t bias, uint8_t shift_s) {
    uint32_t w0 = (uint32_t)(uint8_t)weights[0]
                | ((uint32_t)(uint8_t)weights[1] << 8)
                | ((uint32_t)(uint8_t)weights[2] << 16)
                | ((uint32_t)(uint8_t)weights[3] << 24);

    uint32_t w1 = (uint32_t)(uint8_t)weights[4]
                | ((uint32_t)(uint8_t)weights[5] << 8)
                | ((uint32_t)(uint8_t)weights[6] << 16)
                | ((uint32_t)(uint8_t)weights[7] << 24);

    uint32_t w2 = (uint32_t)(uint8_t)weights[8];

    MAC_WEIGHTS_0 = w0;
    MAC_WEIGHTS_1 = w1;
    MAC_WEIGHTS_2 = w2;
    MAC_BIAS      = (uint32_t)(uint16_t)bias;
    MAC_SHIFT     = (uint32_t)(shift_s & 0x1F);
}

static void load_dense_weights(void) {
    volatile uint32_t *w_mem = (volatile uint32_t *)CLASS_WEIGHTS_BASE;
    for (int i = 0; i < NUM_DENSE_WEIGHTS; i++) {
        int n = i / 196;
        int p = i % 196;
        int offset = (n * 256) + p;
        int8_t w = (int8_t)uart_getchar();
        w_mem[offset] = (uint32_t)(uint8_t)w;
    }
    for (int i = 0; i < NUM_DENSE_BIASES; i++) {
        uint8_t b_lo = (uint8_t)uart_getchar();
        uint8_t b_hi = (uint8_t)uart_getchar();
        int16_t b = (int16_t)((uint16_t)b_hi << 8 | b_lo);
        int offset = 2560 + i;
        w_mem[offset] = (uint32_t)(uint16_t)b;
    }
}

int main(void) {
    display_hex(0x00000000);

    while (1) {
        char token = uart_getchar();
        if (token != 'S') continue;

        display_hex(0x11111111);


        static bool weights_loaded = false;
        if (!weights_loaded) {

            int8_t  weights[KERNEL_TAPS];
            int16_t bias;
            uint8_t shift_s;

            for (int i = 0; i < KERNEL_TAPS; i++)
                weights[i] = (int8_t)uart_getchar();

            uint8_t bias_lo = (uint8_t)uart_getchar();
            uint8_t bias_hi = (uint8_t)uart_getchar();
            bias    = (int16_t)((uint16_t)bias_hi << 8 | bias_lo);
            shift_s = (uint8_t)uart_getchar();

            load_weights(weights, bias, shift_s);
            load_dense_weights();
            weights_loaded = true;
        }

        display_hex(0x33333333);

        uint32_t uart_start_time = get_cycles();


        volatile uint8_t *img = (volatile uint8_t *)IMG_BRAM_BASE;
        for (int i = 0; i < IMAGE_SIZE; i++)
            img[i] = (uint8_t)uart_getchar();

        uint32_t uart_end_time = get_cycles();

        display_hex(0x44444444);


        MAC_IMG_BASE = 0;
        MAC_IMG_LEN = IMAGE_SIZE;

        uint32_t acc_start_time = get_cycles();


        MAC_CTRL = MAC_CTRL_ENABLE;


        while (1) {
            if (MAC_CTRL & MAC_CTRL_RESULT_RDY) {
                uint32_t class_id = OUT_RESULT;
                uint32_t acc_end_time = get_cycles();

                uart_send_byte('R');
                uart_send_byte((uint8_t)(class_id & 0xFF));

                break;
            }
        }


        MAC_CTRL = 0;

        display_hex(0x55555555);
    }

    return 0;
}
