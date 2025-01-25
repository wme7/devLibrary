#ifndef COMMON_H
#define COMMON_H

#include <algorithm>
#include <cassert>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#ifdef __USE_HIP__
#include <hip/hip_runtime.h>
#define devError_t hipError_t
#define devSuccess hipSuccess
#define devStream_t hipStream_t
#define devStreamDefault hipStreamDefault
#define devEvent_t hipEvent_t
#define devGetLastError hipGetLastError
#define devGetErrorString hipGetErrorString
#define devDeviceSynchronize hipDeviceSynchronize
#define devMalloc hipMalloc
#define devFree hipFree
#define devMemcpy hipMemcpy
#define devMemcpyHostToDevice hipMemcpyHostToDevice
#define devMemcpyDeviceToHost hipMemcpyDeviceToHost
#define devMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#else
#include <cuda_runtime.h>
#define devError_t cudaError_t
#define devSuccess cudaSuccess
#define devStream_t cudaStream_t
#define devStreamDefault cudaStreamDefault
#define devEvent_t cudaEvent_t
#define devGetLastError cudaGetLastError
#define devGetErrorString cudaGetErrorString
#define devDeviceSynchronize cudaDeviceSynchronize
#define devMalloc cudaMalloc
#define devFree cudaFree
#define devMemcpy cudaMemcpy
#define devMemcpyHostToDevice cudaMemcpyHostToDevice
#define devMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define devMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
#endif

constexpr int error_exit_code = -1;

#define DEV_CHECK(condition)                                                \
{                                                                           \
    const devError_t error = condition;                                     \
    if (error != devSuccess)                                                \
    {                                                                       \
        std::cerr << "An error encountered: \"" << devGetErrorString(error) \
            << "\" at " << __FILE__ << ':' << __LINE__ << std::endl;        \
        std::exit(error_exit_code);                                         \
    }                                                                       \
}

template<typename T, typename U, 
    std::enable_if_t<std::is_integral<T>::value && std::is_unsigned<U>::value, int> = 0>
__host__ __device__ constexpr unsigned int ceiling_div(const T& dividend, const U& divisor)
{
    return (dividend + divisor - 1) / divisor;
}

#endif // COMMON_H
