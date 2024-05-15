# Fast and Practical Strassen's Matrix Multiplication using FPGAs

This repository contains the source code for the paper "Fast and Practical Strassen's Matrix Multiplication using FPGAs".

TLDR: We present an FPGA-based Strassen's matrix multiplication accelerator that is up to 1.85x faster than the Vitis BLAS L2 GeMM.

## Requirements
- Xilinx Alveo U50/U280 FPGA (If running on board)
- Vitis 2023.2 Installed
- XRT 2023.2 Installed
- Development Target Platform for U50/U280 (Tested on gen3x16 202210)
- Deployment Target Platform for U50/U280 (Tested on gen3x16 2023.2) (If running on board)

The code should work on U250 and U200 as well but has not been tested.

## How to run
1. Clone the repository and navigate to `tests` directory.
```bash
cd L2/tests/gemm_1CU
```

2. If running on board and making the FPGA bitstream from scratch, run the following command (Might take hours):
```bash
make all TARGET=hw PLATFORM=<path/to/xilinx_u50_gen3x16_xdma_5_202210_1.xpfm>
```
This will build both the FPGA bitstream and the host executable.

3. If running on board and using the pre-built FPGA bitstream provided in this repository, first modify the `params.mk` file to use the datatype you desire. Please see the `params.mk` file for more information. After the data type is set as needed, run the following command to build the host executable:
```bash
make host TARGET=hw PLATFORM=<path/to/xilinx_u50_gen3x16_xdma_5_202210_1.xpfm>
```
This will only build the host executable and is fast.

4. For running `sw_emu` or `hw_emu`, run the following command.
```bash
make run TARGET=<sw_emu/hw_emu> PLATFORM=<path/to/xilinx_u50_gen3x16_xdma_5_202210_1.xpfm>
```

By default, the `Makefile` is using the `conn_u50.cfg` file in the `gemm_1CU` directory which is using HBM[0:7]. The file name and its contents can be modified according to the user's needs. E.g., If running on U280, then subtitute `conn_u50.cfg` with `conn_u280.cfg` in `Makefile` and modify the contents of `conn_u280.cfg` to use HBM[0:7] or DDR[0] as needed.

After make is successful, the `gemm_1CU` directory will contain two new directories `_x_temp.hw.xilinx_u50_gen3x16_xdma_5_202210_1` and `build_dir.hw.xilinx_u50_gen3x16_xdma_5_202210_1`. 

The former contains the implemented design which can be opened in vivado (Search for the `*.xpr` file in the directory). 

The latter contains the host executable in case only the host was built (Using step 3), or both the host executable and the FPGA bitstream in case the FPGA bitstream was built (In step 2). The host executable is named `api_gemm.exe` and the FPGA bitstream is named `blasKernel.xclbin`. The file `blasKernel.xclbin.info` contains some details about the implemented kernel such as the memory interfaces it is using and its achieved frequency.

To run the host/FPGA-binary, go to the build directory (i.e., one with the prefix `build_dir.`) and run the following:
```bash
./api_gemm.exe blasKernel.xclbin 2048 2048 2048 2048 2048 2048 2048 1 0
```
This will run the kernel for a 2048x2048 matrix multiplication. The outputs show the kernel workload in clock cycles, time taken in ms to run the kernel, and the achieved TOPS.

A simple bash script is provided to benchmark the built kernel for various matrix sizes evaluated in the paper.
```bash
bash benchmark_kernel.sh <path/to/api_gemm.exe> <path/to/blasKernel.xclbin> <output_csv_name>
```
This will output a csv file with various matrix sizes and their performance stats.

## Pre-built FPGA Bitstreams
We provide pre-built FPGA bitstreams evaluated in the paper. Please navigate to the `prebuilt_bitstreams` directory to find these for U50 and U280. Please note that the host executable has to be compiled by the user on their own machine. Check the `blas.xclbin.info` files for each of the bitstreams to know the HBM/DDR banks used for that bitstream. We provide the following bitstreams:
- U50: `int8`, `int16`, and `int32` bitstreams using HBM[0:7]
- U280: `int8`, `int16`, and `int32` bitstreams using HBM[0:7]
- U280: `int8` and `int16` bitstreams using DDR[0]

Update 10/05/2024: We have added bitstreams for 400 MHz target frequency builds. The frequency achieved by Xilinx's tools are 362 MHz and 371 MHz for the int8 datatype on U50 for Strassen's and baseline kernel, respectively. Detailed exploration of frequency and performance headroom would require utilizing advanced physical design optimizations using frameworks such as AutoBridge and RapidStream that perform better place-and-route for multi-die FPGAs.

## Source Code Structure
The source code is structured as follows:
- `L1` contains the source code for the L1 operations (e.g., L1 GeMM, transpose, double buffer)
- `L2` contains the source code for both the baseline Vitis L2 GeMM kernel and our Strassen's Squared kernel. The kernel codes are located in `L2/include/hw/xf_blas/` directory. L2 modules use L1 wrappers to perform their tasks.
- Host code is located in `L2/src/sw/` directory and its includes are in `L2/include/sw/` directory.
- `prebuilt_bitstreams` contains the pre-built FPGA bitstreams evaluated in the paper. The bitstreams have the file name `blasKernel.xclbin` and can be executed using the host executable that the user has to compile on their own machine using step 3 above.

## Power Consumption
Strassen's and baseline kernel's power consumption are as below. In the paper, we quote the dynamic power minus the power consumed by the GTY transcievers as the transcievers are not used and can be turned off. GTY transcievers consume the same power in both designs, so the results are not impacted. The figures are obtained from post-implementation results of the two kernels from Vivado.

Strassen's Kernel:

![Power Strassen's](https://github.com/afzalxo/FFGeMM/blob/master/images/power-strassens.png?raw=true)

Baseline Kernel:

![Power Vitis GEMM](https://github.com/afzalxo/FFGeMM/blob/master/images/power-baseline.png?raw=true)

## Acknowledgements
To be populated.
