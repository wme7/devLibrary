#!bin/sh

# Configure HIP to use ROCM tools
# export HIP_COMPILER=nvcc
# export HIP_PLATFORM=nvidia
# export HIP_RUNTIME=cuda

# Load ROCM HIP compiler and library
# export PATH="/opt/rocm/bin:$PATH"
# export LD_LIBRARY_PATH="/opt/rocm/lib:/opt/rocm/lib64:$LD_LIBRARY_PATH"

# Manual compilation
# hipcc -shared -o hip_sampleLib.so hip_sampleLib.hip.cpp -fPIC -I/opt/rocm/hipblas/include/ -L/opt/rocm/hipblas/lib -lhipblas
# hipcc -L/path/to/hip_sampleLib.so -o test_sampleLib test_sampleLib.cpp -lhip_sampleLib

cmake ..
