#ifndef DEVICE_COMMON_H
#define DEVICE_COMMON_H

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
#define devGetErrorString hipGetErrorString
#define devEvent_t hipEvent_t
#define devEventCreate hipEventCreate
#define devEventRecord hipEventRecord
#define devEventSynchronize hipEventSynchronize
#define devEventElapsedTime hipEventElapsedTime
#define devEventDestroy hipEventDestroy
#else
#include <cuda_runtime.h>
#define devError_t cudaError_t
#define devSuccess cudaSuccess
#define devGetErrorString cudaGetErrorString
#define devEvent_t cudaEvent_t
#define devEventCreate cudaEventCreate
#define devEventRecord cudaEventRecord
#define devEventSynchronize cudaEventSynchronize
#define devEventElapsedTime cudaEventElapsedTime
#define devEventDestroy cudaEventDestroy
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

#endif // DEVICE_COMMON_H