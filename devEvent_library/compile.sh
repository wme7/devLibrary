#/bin/bash

# specify the HIP_PALTFORM
export HIP_PLATFORM="nvidia"

# Build the project for HIP
mkdir build && cd build
cmake -DBUILD_GPU_LANGUAGE=HIP ..
make

# Build the project for CUDA
mkdir build && cd build
cmake -DBUILD_GPU_LANGUAGE=CUDA ..
make

# Testing
ctest