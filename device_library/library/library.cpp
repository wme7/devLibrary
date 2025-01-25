#include <iostream>
#include <numeric>
#include <vector>
#include "common.h"

// Copy Kernel
__global__ void copy_kernel(uint32_t* out, const uint32_t* in, const unsigned int size)
{
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < size) out[tid] = in[tid];
}

// library function
int foo()
{
    // The number of elements in the input vector.
    constexpr unsigned int size = 512;

    // The number of bytes to allocate for the input- and output device vectors.
    constexpr size_t size_bytes = size * sizeof(uint32_t);

    // The number of threads per kernel block.
    constexpr unsigned int block_size = 256;

    // The number of blocks per kernel grid. The expression below calculates `ceil(size / block_size)`.
    constexpr unsigned int grid_size = ceiling_div(size, block_size);

    // Allocate host input vector and fill it with an increasing sequence (i.e. 0, 1, 2, ...).
    std::vector<uint32_t> in(size);
    std::iota(in.begin(), in.end(), 0);

    // Allocate input and output device vector and copy the input data.
    uint32_t* d_in{};
    uint32_t* d_out{};
    DEV_CHECK(devMalloc(&d_in, size_bytes));
    DEV_CHECK(devMalloc(&d_out, size_bytes));
    DEV_CHECK(devMemcpy(d_in, in.data(), size_bytes, devMemcpyHostToDevice));

    // Launch the kernel on the default stream.
    copy_kernel<<<dim3(grid_size), dim3(block_size), 0, devStreamDefault>>>(d_out, d_in, size);

    // Check if the kernel launch was successful.
    DEV_CHECK(devGetLastError());

    // Copy the results back to the host. This call blocks the host's execution until the copy is finished.
    std::vector<uint32_t> out(size);
    DEV_CHECK(devMemcpy(out.data(), d_out, size_bytes, devMemcpyDeviceToHost));

    // Free device memory.
    DEV_CHECK(devFree(d_in));
    DEV_CHECK(devFree(d_out));

    // Check the results' validity.
    size_t errors = 0;
    for(size_t i = 0; i < size; ++i)
    {
        if(in[i] != out[i])
        {
            ++errors;
        }
    }

    if(errors != 0)
    {
        std::cout << "Validation failed. Errors: " << errors << std::endl;
        // Return control flow to the main program.
        return error_exit_code;
    }
    else
    {
        std::cout << "Validation passed." << std::endl;
    }

    // Return control flow to the main program.
    return 0;
}
