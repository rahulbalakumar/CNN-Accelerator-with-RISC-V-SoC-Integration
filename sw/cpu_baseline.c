

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

static inline uint32_t get_cycles(void) {
    uint32_t cycles;
    __asm__ volatile ("rdcycle %0" : "=r" (cycles));
    return cycles;
}

uint8_t image[IMAGE_SIZE];


int8_t dense_weights[10][196];
int16_t dense_biases[10];

int8_t conv_weights[9];
int16_t conv_bias;
uint8_t conv_shift_s;

void load_model_data() {
    for (int i = 0; i < 9; i++) conv_weights[i] = (int8_t)uart_getchar();
    uint8_t bias_lo = (uint8_t)uart_getchar();
    uint8_t bias_hi = (uint8_t)uart_getchar();
    conv_bias = (int16_t)((uint16_t)bias_hi << 8 | bias_lo);
    conv_shift_s = (uint8_t)uart_getchar();

    for (int i = 0; i < NUM_DENSE_WEIGHTS; i++) {
        int n = i / 196;
        int p = i % 196;
        dense_weights[n][p] = (int8_t)uart_getchar();
    }
    for (int i = 0; i < NUM_DENSE_BIASES; i++) {
        uint8_t b_lo = (uint8_t)uart_getchar();
        uint8_t b_hi = (uint8_t)uart_getchar();
        dense_biases[i] = (int16_t)((uint16_t)b_hi << 8 | b_lo);
    }
}

int main() {
    while (1) {
        char token = uart_getchar();
        if (token != 'S') continue;

        static bool loaded = false;
        if (!loaded) {
            load_model_data();
            loaded = true;
        }

        for (int i = 0; i < IMAGE_SIZE; i++) {
            image[i] = uart_getchar();
        }

        uint32_t start_cycles = get_cycles();

        int8_t pooled[196];
        int pool_idx = 0;

        for (int r = 0; r < 28; r += 2) {
            for (int c = 0; c < 28; c += 2) {
                int8_t max_val = -128;
                for (int wr = 0; wr < 2; wr++) {
                    for (int wc = 0; wc < 2; wc++) {
                        int pr = r + wr;
                        int pc = c + wc;

                        int32_t acc = conv_bias;
                        for (int kr = -1; kr <= 1; kr++) {
                            for (int kc = -1; kc <= 1; kc++) {
                                int ir = pr + kr;
                                int ic = pc + kc;
                                int16_t pix = 0;
                                if (ir >= 0 && ir < 28 && ic >= 0 && ic < 28) {
                                    pix = image[ir * 28 + ic];
                                }
                                int8_t w = conv_weights[(kr+1)*3 + (kc+1)];
                                acc += pix * w;
                            }
                        }

                        acc = acc >> conv_shift_s;

                        if (acc > 127) acc = 127;
                        if (acc < -128) acc = -128;

                        int8_t relu = (acc > 0) ? (int8_t)acc : 0;

                        if (relu > max_val) max_val = relu;
                    }
                }
                pooled[pool_idx++] = max_val;
            }
        }

        int best_class = 0;
        int32_t best_score = -2147483648;

        for (int i = 0; i < 10; i++) {
            int32_t score = dense_biases[i];
            for (int j = 0; j < 196; j++) {
                score += (int16_t)pooled[j] * (int16_t)dense_weights[i][j];
            }
            if (score > best_score) {
                best_score = score;
                best_class = i;
            }
        }

        uint32_t end_cycles = get_cycles();
        uint32_t total_cycles = end_cycles - start_cycles;

        uart_send_byte('R');
        uart_send_byte((uint8_t)(best_class & 0xFF));
    }
    return 0;
}
