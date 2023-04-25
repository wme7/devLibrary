#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "cuda_sampleLib.h"

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


namespace cuda {

template <class T>
void memory<T>::allocate(T** dev_mem, size_t N)
{
  cudaMalloc(dev_mem, N * sizeof(T));
}

template <class T>
void memory<T>::free(T* dev_mem) { 
  cudaFree(dev_mem); 
}

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
  cudaDeviceSynchronize();
}

// explicit instantiations
template class memory<float>;
template class memory<double>;
template class blas<float>;
template class blas<double>;
template class display<float>;
template class display<double>;

} // namespace cuda
