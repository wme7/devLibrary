# Device Library
Writen by Manuel A. Diaz on 25.01.2025

This library contains two examples of C++ libraries that contain GPU-functionalities. These libraries are a proof-of-concept for building cross-platform libraries that can be explicitly build on either CUDA- or HIP-runtime.

## Introduction
In this example we aim to response a simple question:

> How to build single cross-platform library with `CMake` that builds on either NVIDIA or AMD platforms?

## Exploration path:

 - A initial CUDA version of the library to be used as a study case. 
  We refer to it from here on as the *cuda library*.
 - A *hipified* version of the *cuda library* is then explored.
  We aim to build it on AMD and NVidia platforms without any special treatment, we refer to it as the *hip library*.
 - An explicit layer to swap for the functionalities of both runtimes is build on a common header. 
 - We explore CMake 3.21, as HIP has become a compilation language to control the sources language property.

## Results:

 - The library compiled with CUDA language builds and runs well on NVIDIA platforms.
 - The library compiled with HIP language builds and runs well on AMD platforms.
  
## Discussion:

HIP 6.2.4 finally seems to working according to expectations! Although we had to wait for very long to reach this point. It seems that cross-platform gpu-libraries can be finally build in a systematic manner with `CMake`.

We have tested earlier versions of HIP (5.3, 5.6 and 6.1) with their companion hipBLAS libries and noticed this was not a posibility. In these earlier versions, the expectation of developers were to `make` the compilation of any development almost by hand. Thus, a directly development of a cross-platform would have required a third-party library like [kokkos](https://github.com/kokkos/kokkos).

## About the Repo:
This repo is organized as follows:
```bash
.
├── devBLAS_library
│   ├── CMakeLists.txt
│   ├── compile.sh
│   ├── library
│   │   ├── common.h
│   │   ├── library.cpp
│   │   └── library.h
│   └── main.cpp
├── device_library
│   ├── CMakeLists.txt
│   ├── compile.sh
│   ├── library
│   │   ├── common.h
│   │   ├── library.cpp
│   │   └── library.h
│   └── main.cpp
├── LICENSE
└── README.md
```
where the scripts `compile.sh` contains the details used to build the libraries.

## Build
Assuming that the `CUDAtoolkit` 12.6 and the `HIP` 6.2.4 [package](https://github.com/ROCm-Developer-Tools/HIP) have been properly installed/built on a ubuntu-linux 24.04 box, each sample_library should be build using `CMake` > 3.21 as:
```bash 
$ mkdir build && cd build
$ cmake -DBUILD_GPU_LANGUAGE=HIP ..
$ make
```
Typical output on a **NVIDIA platform** reads :
```bash
$ cmake -DBUILD_GPU_LANGUAGE=CUDA ..
-- The CXX compiler identification is GNU 13.3.0
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /usr/bin/c++ - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- The CUDA compiler identification is NVIDIA 12.6.85
-- Detecting CUDA compiler ABI info
-- Detecting CUDA compiler ABI info - done
-- Check for working CUDA compiler: /usr/local/cuda-12.6/bin/nvcc - skipped
-- Detecting CUDA compile features
-- Detecting CUDA compile features - done
-- Found CUDAToolkit: /usr/local/cuda-12.6/targets/x86_64-linux/include (found version "12.6.85") 
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD - Success
-- Found Threads: TRUE  
-- Configuring done (1.7s)
-- Generating done (0.0s)
-- Build files have been written to: /home/mdiaz/Depots/devLibrary/devBLAS_library/build

$ make
[ 16%] Building CUDA object CMakeFiles/Test.dir/library/library.cpp.o
[ 33%] Linking CUDA shared library libTest.so
[ 33%] Built target Test
[ 50%] Building CXX object CMakeFiles/devBLAS_library_clang.dir/main.cpp.o
[ 66%] Linking CXX executable devBLAS_library_clang
[ 66%] Built target devBLAS_library_clang
[ 83%] Building CXX object CMakeFiles/devBLAS_library_cxx.dir/main.cpp.o
[100%] Linking CXX executable devBLAS_library_cxx
[100%] Built target devBLAS_library_cxx
```

Typical output on a **AMD platform** reads :
```bash
$ cmake -DBUILD_GPU_LANGUAGE=HIP ..
-- The CXX compiler identification is Clang 14.0.0
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /opt/rocm/bin/hipcc - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Found HIP: /opt/rocm (found version "6.2.1153-02187ecf") 
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
[ 16%] Building HIP object CMakeFiles/Test.dir/library/library.cpp.o
[ 33%] Linking HIP shared library libTest.so
[ 33%] Built target Test
[ 50%] Building CXX object CMakeFiles/devBLAS_library_clang.dir/main.cpp.o
[ 66%] Linking CXX executable devBLAS_library_clang
[ 66%] Built target devBLAS_library_clang
[ 83%] Building CXX object CMakeFiles/devBLAS_library_cxx.dir/main.cpp.o
[100%] Linking CXX executable devBLAS_library_cxx
[100%] Built target devBLAS_library_cxx
```

## What's the trick on *device library*?

No magic, no work-arounds, straitforwardly:
* Use CMake's feature to use HIP as a compilation language
* A header to encapsulate the HIP or CUDA runtime functionalities used in the library.

## Test

For each case library, the main in build using AMD's clang compiler or the Host's gcc compiler. On both scenarios we obtained:

```bash
$ ctest
Test project /home/mdiaz/Depots/devLibrary/device_library/build
    Start 1: device_library_clang
1/2 Test #1: device_library_clang .............   Passed    0.36 sec
    Start 2: device_library_cxx
2/2 Test #2: device_library_cxx ...............   Passed    0.23 sec

100% tests passed, 0 tests failed out of 2

Total Test time (real) =   0.59 sec


$ ctest
Test project /path/to/devLibrary/devBLAS_library/build
    Start 1: devBLAS_library_clang
1/2 Test #1: devBLAS_library_clang ............   Passed    0.56 sec
    Start 2: devBLAS_library_cxx
2/2 Test #2: devBLAS_library_cxx ..............   Passed    0.27 sec

100% tests passed, 0 tests failed out of 2

Total Test time (real) =   0.83 sec
```

## Disclaimer
I have tested this only on:
  - Personal Computer with NVIDIA QUADRO ( **NVIDIA platform** )
  - The supercomputer of the north: [LUMI](https://www.lumi-supercomputer.eu/may-we-introduce-lumi/) nodes ( **AMD platform** )

happy coding !
 - M.D.