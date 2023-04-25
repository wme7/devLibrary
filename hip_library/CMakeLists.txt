cmake_minimum_required(VERSION 3.16)

project(myPrototypes VERSION 1.0.0 LANGUAGES CXX)

# Set compilation standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find HIP
list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)
find_package(HIP REQUIRED)

# Build HIP depending on platform
if(${HIP_PLATFORM} MATCHES "nvidia")
  message(STATUS "HIP_PLATFORM: nvidia")
  add_compile_definitions(__HIP_PLATFORM_NVIDIA__)
  find_package(CUDAToolkit REQUIRED)
  set(CMAKE_CUDA_ARCHITECTURES "70;75")
  set(DEPENDENCIES CUDA::cudart CUDA::cublas)
else()
  message(STATUS "HIP_PLATFORM: amd")
  add_compile_definitions(__HIP_PLATFORM_AMD__)
  set(CMAKE_HIP_ARCHITECTURES "gfx906;gfx908")
  find_package(hipblas REQUIRED)
  set(DEPENDENCIES roc::hipblas)
endif()

# Library:
add_library(hip_sampleLib SHARED hip_sampleLib.hip.cpp hip_sampleLib.h)
target_link_libraries(hip_sampleLib PRIVATE ${DEPENDENCIES})

# Executables:
add_executable(test_sampleLib.run test_sampleLib.cpp)
target_link_libraries(test_sampleLib.run hip_sampleLib)
