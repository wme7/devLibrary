cmake_minimum_required(VERSION 3.16)

project(myPrototypes VERSION 1.0.0 LANGUAGES CXX)

# Define ROCM_PATH if not defined
if (NOT DEFINED ROCM_PATH)
  set(ROCM_PATH "/opt/rocm" CACHE STRING "Default ROCM installation directory.")
endif()

# Search for rocm in common locations
list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}/hip" "${ROCM_PATH}")
set(CMAKE_MODULE_PATH "${ROCM_PATH}/hip/cmake")

# Find hip
find_package(HIP MODULE REQUIRED)
find_package(hipblas REQUIRED)

# Generate a compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set compilation standard
set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})
set(CMAKE_CXX_LINKER   ${HIP_HIPCC_EXECUTABLE})
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Library:
add_library(hip_sampleLib SHARED hip_sampleLib.hip.cpp hip_sampleLib.h)
target_link_libraries(hip_sampleLib PRIVATE roc::hipblas)

# Executables:
add_executable(test_sampleLib.run test_sampleLib.cpp)
target_link_libraries(test_sampleLib.run hip_sampleLib)
