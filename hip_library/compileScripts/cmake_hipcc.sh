#!bin/sh
cmake \
  -DCMAKE_MODULE_PATH=/opt/rocm/hip/cmake \
  -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  ..
  