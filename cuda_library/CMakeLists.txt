cmake_minimum_required(VERSION 3.16)

project(myPrototypes VERSION 1.0.0 LANGUAGES CXX CUDA)

# Set compilation standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "70;75")

# Find CUDAToolkit
find_package(CUDAToolkit REQUIRED)
# link_libraries(CUDA::cudart CUDA::cublas)
# link_libraries(CUDA::thrust INTERFACE IMPORTED)

# Library
add_library(cuda_sampleLib SHARED cuda_sampleLib.cu cuda_sampleLib.h)
target_link_libraries(cuda_sampleLib PRIVATE CUDA::cudart CUDA::cublas)

# Executables:
add_executable(test_sampleLib.run test_sampleLib.cpp)
target_link_libraries(test_sampleLib.run cuda_sampleLib)
