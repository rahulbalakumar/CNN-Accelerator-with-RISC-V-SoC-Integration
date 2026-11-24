# cnn_soc_top.sdc - timing constraints for the DE2-115 build (50 MHz)
# Add to the project:  set_global_assignment -name SDC_FILE cnn_soc_top.sdc

create_clock -name CLOCK_50 -period 20.000 [get_ports {CLOCK_50}]
derive_clock_uncertainty

# Asynchronous inputs: the push buttons and the UART line are not clocked by CLOCK_50.
# The UART RX goes through a 2-flop synchroniser inside uart_serial_controller.
set_false_path -from [get_ports {KEY[*] UART_RXD}]

# Slow outputs (7-segment displays, UART TX): no timing relationship to an external clock.
set_false_path -to [get_ports {HEX0[*] HEX1[*] HEX2[*] HEX3[*] HEX4[*] HEX5[*] HEX6[*] HEX7[*] UART_TXD}]
