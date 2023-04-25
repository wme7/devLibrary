#include "device_sampleLib.h"
#include "hip/hip_runtime.h"
#ifdef __HIP_PLATFORM_AMD__
    #include "hipblas/hipblas.h"
#else
    #include <cublas_v2.h>
#endif

namespace host {

template <class T>
void memory<T>::allocate(T** host_mem, size_t N)
{
  *host_mem = new T[N]; 
}

template <class T>
void memory<T>::free(T* host_mem) { 
  delete[] host_mem;
}

// explicit instantiations
template class memory<float>;
template class memory<double>;

} // namespace host


namespace device {

template <class T>
void memory<T>::allocate(T** dev_mem, size_t N)
{
  hipMalloc(dev_mem, N * sizeof(T));
}

template <class T>
void memory<T>::free(T* dev_mem) { 
  hipFree(dev_mem); 
}

#ifdef __HIP_PLATFORM_AMD__

template <class T>
void blas<T>::setVector(int n, const T *x, int incx, T *y, int incy) 
{
  hipblasSetVector(n, sizeof(T), x, incx, y, incy);
}

template <class T>
void blas<T>::getVector(int n, const T *x, int incx, T *y, int incy) 
{
  hipblasGetVector(n, sizeof(T), x, incx, y, incy);
}

#else

template <class T>
void blas<T>::setVector(int n, const T *x, int incx, T *y, int incy) 
{
  cublasSetVector(n, sizeof(T), x, incx, y, incy);
}

template <class T>
void blas<T>::getVector(int n, const T *x, int incx, T *y, int incy) 
{
  cublasGetVector(n, sizeof(T), x, incx, y, incy);
}

#endif

namespace kernel {
template <class T>
__global__ void printVector(const T* array, size_t size)
{
  printf(" -- print vector -- \n");
  for (size_t i=0; i<size; i++) {
    printf("%7.4f, ",array[i]);
  } printf("\n\n");
}
} // namespace kernel

template <class T>
void display<T>::printVector(const T* array, size_t size)
{
  kernel::printVector<T><<<1, 1>>>(array, size);
  hipDeviceSynchronize();
}

// explicit instantiations
template class memory<float>;
template class memory<double>;
template class blas<float>;
template class blas<double>;
template class display<float>;
template class display<double>;

} // namespace device
