# Full-SoC MNIST test in ModelSim / Questa.
# In the ModelSim console:  cd <repo>/tb/system   then   do run_modelsim.do
vlib work
vlog -sv -f files.f
vsim -c -voptargs=+acc work.tb_soc_mnist
run -all
quit -f
