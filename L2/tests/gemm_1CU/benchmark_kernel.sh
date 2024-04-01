#!/bin/bash
#
# Benchmark the kernel for all matrix sizes
#
# Usage: benchmark_kernel.sh <host.exe> <kernel.xclbin> <output_file>

# create an array of matrix sizes
matrix_sizes = (256, 512, 768, 1024, 1536, 2048, 2560, 3072, 4096, 5120, 6144, 7168, 8192, 9216, 10240, 11264)

host_binary = $1
kernel_xclbin = $2
output_file = $3

# run the kernel for each matrix size
# and record the stdout and stderr to the output file
for size in ${matrix_sizes[@]}; do
    echo "Running kernel for matrix size $size"
    $host_binary $kernel_xclbin $size $size $size $size $size $size $size 1 0 2>&1 | tee -a $output_file
done
