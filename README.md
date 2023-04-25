# devLibrary
A sample library to test cross-platform capabilities of `HIP` API and `roc::hipblas` library.

Writen by Manuel A. Diaz @ CENAERO ASBL on 25.04.2023 

## Contents of this repo:
This repo contains three version of as single sample-library in C++17, namely:
 - A initial CUDA version of the library to be used as a study case. 
  We refer to it from here on as the *cuda library*.
 - A manually hipified version of the *cuda library*.
  Because it is meant to be built on AMD platforms platforms without any special treatment, we refer to it as the *hip library*.
 - A cross-platform ( NVIDIA / AMD ) version of the *cuda library*. 
  We refere to it as the *device library* because it builds on NVIDIA and/or AMD platforms with a special trick to overcomes the compilation problem observed hipblas library. See forum discussions [(1)](https://www.reddit.com/r/ROCm/comments/12bmygw/how_do_you_build_apps_with_hipblas_using_cmake/) and [(2)](https://www.reddit.com/r/cmake/comments/12iknc9/building_crossplataform_libraries_with_hip_in_c/).

This repo is organized as follows:
```bash
.
├── cuda_library
│   ├── CMakeLists.txt
│   ├── compileScripts
│   │   └── cmake_cxx_cuda.sh
│   ├── cuda_sampleLib.cu
│   ├── cuda_sampleLib.h
│   └── test_sampleLib.cpp
├── device_library
│   ├── CMakeLists.txt
│   ├── compileScripts
│   │   └── cmake_hipcc.sh
│   ├── device_sampleLib.h
│   ├── device_sampleLib.hip.cpp
│   └── test_sampleLib.cpp
├── hip_library
│   ├── CMakeLists.txt
│   ├── compileScripts
│   │   └── cmake_hipcc.sh
│   ├── hip_sampleLib.h
│   ├── hip_sampleLib.hip.cpp
│   └── test_sampleLib.cpp
├── LICENSE
└── README.md
```
## Build
Assuming that the `CUDAtoolkit` and the `HIP` [package](https://github.com/ROCm-Developer-Tools/HIP) have been properly installed/built on a linux box, each sample_library can be build using `CMake` > 3.16 by sourcing its respetive compilation script:
```bash 
$ mkdir build && cd build
$ source ../compileScripts/cmake_*.sh
```
Typical output on a **NVIDIA platform** reads :
```bash
$ source ../compileScripts/cmake_hipcc.sh 
-- The CXX compiler identification is GNU 8.5.0
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /opt/rocm/bin/hipcc - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Found HIP: /opt/rocm-5.4.0/hip (found version "5.4.22801-aaa1e3d8") 
-- HIP_PLATFORM: nvidia
-- Found CUDAToolkit: /usr/local/cuda-11.8/include (found version "11.8.89") 
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD - Success
-- Found Threads: TRUE  
-- Configuring done (1.6s)
-- Generating done (0.0s)
-- Build files have been written to: /home/mdiaz/Depots/devLibrary/device_library/build

$ make
[ 25%] Building CUDA object CMakeFiles/cuda_sampleLib.dir/cuda_sampleLib.cu.o
[ 50%] Linking CUDA shared library libcuda_sampleLib.so
[ 50%] Built target cuda_sampleLib
[ 75%] Building CXX object CMakeFiles/test_sampleLib.run.dir/test_sampleLib.cpp.o
[100%] Linking CXX executable test_sampleLib.run
[100%] Built target test_sampleLib.run
```

Typical output on a **AMD platform** reads :
```bash
$ source ../compileScripts/cmake_hipcc.sh 
-- The CXX compiler identification is Clang 14.0.0
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /opt/rocm/bin/hipcc - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Found HIP: /opt/rocm (found version "5.2.21153-02187ecf") 
-- Performing Test HIP_CLANG_SUPPORTS_PARALLEL_JOBS
-- Performing Test HIP_CLANG_SUPPORTS_PARALLEL_JOBS - Success
-- HIP_PLATFORM: amd
-- Looking for C++ include pthread.h
-- Looking for C++ include pthread.h - found
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD - Success
-- Found Threads: TRUE  
-- hip::amdhip64 is SHARED_LIBRARY
-- Configuring done
-- Generating done
-- Build files have been written to: /users/diazesco/Depots/devLibrary/hip_library/build

$ make 
[ 25%] Building CXX object CMakeFiles/device_sampleLib.dir/device_sampleLib.hip.cpp.o
[ 50%] Linking CXX shared library libdevice_sampleLib.so
[ 50%] Built target device_sampleLib
[ 75%] Building CXX object CMakeFiles/test_sampleLib.run.dir/test_sampleLib.cpp.o
[100%] Linking CXX executable test_sampleLib.run
[100%] Built target test_sampleLib.run
```
NOTE: I have excluded all warning output messages from `hipcc` for the sake of clarity. To avoid them, one must implement a `deviceErrChecker()` function/macro. `hipcc` is very picky in this regard.

## What's trick on *device library*?

Given the `CMake` building problem with `hipblas` libraries and `hipcc` on **NVIDIA platform**, see [(1)](https://www.reddit.com/r/ROCm/comments/12bmygw/how_do_you_build_apps_with_hipblas_using_cmake/) and [(2)](https://www.reddit.com/r/cmake/comments/12iknc9/building_crossplataform_libraries_with_hip_in_c/)), a preprocessing switch has been implemented so that the `cublas` library is used when compiling the *device_library* on **NVIDIA platform**s and the `hipblas` library when on **AMD platform**s. 

## Test

Verify that for each case, the library test script produces the following output:

```bash
./test_sampleLib.run 
 -- print vector -- 
 0.0000,  1.0000,  2.0000,  3.0000,  4.0000,  5.0000,  6.0000,  7.0000,  8.0000,  9.0000, 10.0000, 11.0000, 12.0000, 13.0000, 14.0000, 15.0000, 16.0000, 17.0000, 18.0000, 19.0000, 20.0000, 21.0000, 22.0000, 23.0000, 24.0000, 25.0000, 26.0000, 27.0000, 28.0000, 29.0000, 30.0000, 31.0000, 32.0000, 33.0000, 34.0000, 35.0000, 36.0000, 37.0000, 38.0000, 39.0000, 40.0000, 41.0000, 42.0000, 43.0000, 44.0000, 45.0000, 46.0000, 47.0000, 48.0000, 49.0000, 50.0000, 51.0000, 52.0000, 53.0000, 54.0000, 55.0000, 56.0000, 57.0000, 58.0000, 59.0000, 60.0000, 61.0000, 62.0000, 63.0000, 64.0000, 65.0000, 66.0000, 67.0000, 68.0000, 69.0000, 70.0000, 71.0000, 72.0000, 73.0000, 74.0000, 75.0000, 76.0000, 77.0000, 78.0000, 79.0000, 80.0000, 81.0000, 82.0000, 83.0000, 84.0000, 85.0000, 86.0000, 87.0000, 88.0000, 89.0000, 90.0000, 91.0000, 92.0000, 93.0000, 94.0000, 95.0000, 96.0000, 97.0000, 98.0000, 99.0000, 100.0000, 101.0000, 102.0000, 103.0000, 104.0000, 105.0000, 106.0000, 107.0000, 108.0000, 109.0000, 110.0000, 111.0000, 112.0000, 113.0000, 114.0000, 115.0000, 116.0000, 117.0000, 118.0000, 119.0000, 120.0000, 121.0000, 122.0000, 123.0000, 124.0000, 125.0000, 126.0000, 127.0000, 128.0000, 129.0000, 130.0000, 131.0000, 132.0000, 133.0000, 134.0000, 135.0000, 136.0000, 137.0000, 138.0000, 139.0000, 140.0000, 141.0000, 142.0000, 143.0000, 144.0000, 145.0000, 146.0000, 147.0000, 148.0000, 149.0000, 150.0000, 151.0000, 152.0000, 153.0000, 154.0000, 155.0000, 156.0000, 157.0000, 158.0000, 159.0000, 160.0000, 161.0000, 162.0000, 163.0000, 164.0000, 165.0000, 166.0000, 167.0000, 168.0000, 169.0000, 170.0000, 171.0000, 172.0000, 173.0000, 174.0000, 175.0000, 176.0000, 177.0000, 178.0000, 179.0000, 180.0000, 181.0000, 182.0000, 183.0000, 184.0000, 185.0000, 186.0000, 187.0000, 188.0000, 189.0000, 190.0000, 191.0000, 192.0000, 193.0000, 194.0000, 195.0000, 196.0000, 197.0000, 198.0000, 199.0000, 200.0000, 201.0000, 202.0000, 203.0000, 204.0000, 205.0000, 206.0000, 207.0000, 208.0000, 209.0000, 210.0000, 211.0000, 212.0000, 213.0000, 214.0000, 215.0000, 216.0000, 217.0000, 218.0000, 219.0000, 220.0000, 221.0000, 222.0000, 223.0000, 224.0000, 225.0000, 226.0000, 227.0000, 228.0000, 229.0000, 230.0000, 231.0000, 232.0000, 233.0000, 234.0000, 235.0000, 236.0000, 237.0000, 238.0000, 239.0000, 240.0000, 241.0000, 242.0000, 243.0000, 244.0000, 245.0000, 246.0000, 247.0000, 248.0000, 249.0000, 250.0000, 251.0000, 252.0000, 253.0000, 254.0000, 255.0000, 
```

## Disclaimer
I have tested this only on:
  - The supercomputer of Wallonia: [LUCIA](https://tier1.cenaero.be/en/lucia-kickoff) nodes ( **NVIDIA platform** )
  - The supercomputer of the north: [LUMI](https://www.lumi-supercomputer.eu/may-we-introduce-lumi/) nodes ( **AMD platform** )

happy coding !
 - M.D.