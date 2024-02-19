cmake_minimum_required(VERSION 3.16)

project(myPrototypes VERSION 1.0.0 LANGUAGES CXX)

# Define ROCM_PATH if not defined
if (NOT DEFINED ROCM_PATH)
  set(ROCM_PATH "/opt/rocm" CACHE STRING "Default ROCM installation directory.")
endif()

# Search for rocm in common locations
list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}/hip" "${ROCM_PATH}")

# Find hip
if("$ENV{HIP_PLATFORM}" STREQUAL "nvidia")
  set(CMAKE_MODULE_PATH "${ROCM_PATH}/hip/cmake")
  find_package(HIP MODULE REQUIRED)
else()
  find_package(hip)
endif()

# Generate a compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set compilation standard
set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})
set(CMAKE_CXX_LINKER   ${HIP_HIPCC_EXECUTABLE})
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

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
add_library(device_sampleLib SHARED device_sampleLib.hip.cpp device_sampleLib.h)
target_link_libraries(device_sampleLib PRIVATE ${DEPENDENCIES})

# Executables:
add_executable(test_sampleLib.run test_sampleLib.cpp)
target_link_libraries(test_sampleLib.run device_sampleLib)
