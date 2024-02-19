#!bin/sh

# Configure HIP to use NVIDIA toolkit
# export HIP_COMPILER=nvcc
# export HIP_PLATFORM=nvidia
# export HIP_RUNTIME=cuda

# Load NVIDIA toolkit compiler and library
# export PATH="/usr/local/cuda-11.8/bin:$PATH"
# export LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH"

# Load ROCM HIP compiler and library
# export PATH="/opt/rocm/bin:$PATH"
# export LD_LIBRARY_PATH="/opt/rocm/lib:/opt/rocm/lib64:$LD_LIBRARY_PATH"

cmake ..
