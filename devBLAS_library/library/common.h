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
#include <hipblas/hipblas.h>
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
#define devblasStatus_t hipblasStatus_t
#define devblasStatusString hipblasStatusToString
#define DEVBLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS
#define devblasSetVector hipblasSetVector
#define devblasGetVector hipblasGetVector
#else
#include <cuda_runtime.h>
#include <cublas_v2.h>
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
#define devblasStatus_t cublasStatus_t
#define devblasStatusString cublasGetStatusString
#define DEVBLAS_STATUS_SUCCESS CUBLAS_STATUS_SUCCESS
#define devblasSetVector cublasSetVector
#define devblasGetVector cublasGetVector
#endif

constexpr int error_exit_code = -1;

#define DEV_CHECK(condition)                                                 \
{                                                                            \
    const devError_t status = condition;                                     \
    if (status != devSuccess)                                                \
    {                                                                        \
        std::cerr << "An error encountered: \"" << devGetErrorString(status) \
            << "\" at " << __FILE__ << ':' << __LINE__ << std::endl;         \
        std::exit(error_exit_code);                                          \
    }                                                                        \
}

#define DEVBLAS_CHECK(condition)                                                    \
{                                                                                   \
    const devblasStatus_t status = condition;                                       \
    if(status != DEVBLAS_STATUS_SUCCESS)                                            \
    {                                                                               \
        std::cerr << "devBLAS error encountered: \"" << devblasStatusString(status) \
                    << "\" at " << __FILE__ << ':' << __LINE__ << std::endl;        \
        std::exit(error_exit_code);                                                 \
    }                                                                               \
}

#endif // COMMON_H
