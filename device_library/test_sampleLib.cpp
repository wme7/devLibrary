#include "device_sampleLib.h"

using real = float; // or double

int main() {
  // Allocate host and device memory
  int N = 1 << 8; // 256 elements
  real *h_in;    host::memory<real>::allocate(&h_in, N);
  real *d_var; device::memory<real>::allocate(&d_var, N);
  real *h_out;   host::memory<real>::allocate(&h_out, N);

  // Fill the input vector with sequential integers
  for (int i = 0; i < N; ++i) { h_in[i] = i; }

  // Copy the input vector from host to device
  device::blas<real>::setVector(N, h_in, 1, d_var, 1);

  // Launch the kernel
  device::display<real>::printVector(d_var, N);

  // Copy the output vector from device to host
  device::blas<real>::getVector(N, d_var, 1, h_out, 1);

  // Sanity check
  for (int i = 0; i < N; ++i) {
    if (h_in[i] != h_out[i]) {
      printf("Error: h_in[%d] = %f != h_out[%d] = %f", i, h_in[i], i, h_out[i]);
      return 1;
    }
  }

  // Release resources
    host::memory<real>::free(h_in);
  device::memory<real>::free(d_var);
    host::memory<real>::free(h_out);
  return 0;
}