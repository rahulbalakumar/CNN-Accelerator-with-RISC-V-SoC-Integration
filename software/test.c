#include <stdint.h>

#define UART_TX_DATA   (*(volatile uint8_t *)0x40000000)
#define UART_TX_STATUS (*(volatile uint8_t *)0x40000004)

#define ACCEL_BASE_ADDR (*(volatile uint32_t *)0x40000008)
#define ACCEL_WIDTH     (*(volatile uint32_t *)0x400000012)
#define ACCEL_CONTROL   (*(volatile uint32_t *)0x400000016)
#define ACCEL_STATUS    (*(volatile uint32_t *)0x40000020)

void uart_put_char(char c) {
    while(UART_TX_STATUS == 1){ // 1 means it is busy
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
    while (ACCEL_STATUS == 1) { // Busy = 1, ~Busy = 0
        continue;
    }
}