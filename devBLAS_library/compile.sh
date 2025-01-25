#/bin/bash

# specify the HIP_PALTFORM
export HIP_PLATFORM="nvidia"

# Build the project for HIP
mkdir build_CUDA && cd build_HIP
cmake -DBUILD_GPU_LANGUAGE=HIP ..
make

# Build the project for CUDA
mkdir build_CUDA && cd build_CUDA
cmake -DBUILD_GPU_LANGUAGE=CUDA ..
make
