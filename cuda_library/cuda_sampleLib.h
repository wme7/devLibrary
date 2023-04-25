// My "super" library for CUDA
#include <cstdio>

namespace host {

template <typename T>
struct memory
{
  static void allocate(T** host_mem, size_t N);
  static void free(T* host_mem);
};
}  // namespace host


namespace cuda {

template <typename T>
struct memory
{
  static void allocate(T** dev_mem, size_t N);
  static void free(T* dev_mem);
};

template <typename T>
struct blas
{
  static void setVector(int n, const T *x, int incx, T *y, int incy);
  static void getVector(int n, const T *x, int incx, T *y, int incy);
};

template <typename T>
struct display
{
  static void printVector(const T *x, size_t n);
};
}  // namespace cuda