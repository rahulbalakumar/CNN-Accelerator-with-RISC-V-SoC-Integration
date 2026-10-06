


void uart_put_char(char c) {
    while(UART_TX_STATUS == 1){
        continue;
    }
    UART_TX_DATA = c;
}

void uart_put_string(const char *s) {
    while(*s != '\0'){
        uart_put_char(*s);
        s++;
    }
}

void accel_run(uint32_t base_addr, uint32_t width) {
    ACCEL_BASE_ADDR = base_addr;
    ACCEL_WIDTH = width;
    ACCEL_CONTROL = ACCEL_CONTROL | 1;
    while (ACCEL_STATUS == 1) {
        continue;
    }
}
