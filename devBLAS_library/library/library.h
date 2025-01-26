// My "super" library for working on HIP or CUDA

#include <cstdio>

namespace host {

template <class T>
struct memory
{
  static void allocate(T** host_mem, size_t N);
  static void free(T* host_mem);
};

extern template struct memory<float>;
extern template struct memory<double>;
}  // namespace host


namespace device {

template <class T>
struct memory
{
  static void allocate(T** dev_mem, size_t N);
  static void free(T* dev_mem);
};

template <class T>
struct blas
{
  static void setVector(int n, const T *x, int incx, T *y, int incy);
  static void getVector(int n, const T *x, int incx, T *y, int incy);
};

template <class T>
struct display
{
  static void printVector(const T *x, size_t n);
};

extern template struct memory<float>;
extern template struct memory<double>;
extern template struct blas<float>;
extern template struct blas<double>;
extern template struct display<float>;
extern template struct display<double>;
}  // namespace device