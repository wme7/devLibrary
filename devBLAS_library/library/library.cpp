#include "library.h"
#include "common.h"


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
template struct memory<float>;
template struct memory<double>;
} // namespace host


namespace device {

template <class T>
void memory<T>::allocate(T** dev_mem, size_t N)
{
  DEV_CHECK(devMalloc(dev_mem, N * sizeof(T)));
}

template <class T>
void memory<T>::free(T* dev_mem) { 
  DEV_CHECK(devFree(dev_mem));
}

template <class T>
void blas<T>::setVector(int n, const T *x, int incx, T *y, int incy) 
{
  DEVBLAS_CHECK(devblasSetVector(n, sizeof(T), x, incx, y, incy));
}

template <class T>
void blas<T>::getVector(int n, const T *x, int incx, T *y, int incy) 
{
  DEVBLAS_CHECK(devblasGetVector(n, sizeof(T), x, incx, y, incy));
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
  DEV_CHECK(devDeviceSynchronize());
}

// explicit instantiations
template struct memory<float>;
template struct memory<double>;
template struct blas<float>;
template struct blas<double>;
template struct display<float>;
template struct display<double>;
} // namespace device
