cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(devBLAS_library LANGUAGES CXX)

set(BUILD_GPU_LANGUAGE "HIP" CACHE STRING "Switches between HIP and CUDA")
set_property(CACHE BUILD_GPU_LANGUAGE PROPERTY STRINGS "HIP" "CUDA")

enable_language(${BUILD_GPU_LANGUAGE})
set(CMAKE_${BUILD_GPU_LANGUAGE}_STANDARD 17)
set(CMAKE_${BUILD_GPU_LANGUAGE}_EXTENSIONS OFF)
set(CMAKE_${BUILD_GPU_LANGUAGE}_STANDARD_REQUIRED ON)

set(ROCM_ROOT "/opt/rocm" CACHE PATH "Root directory of the ROCm installation")
list(APPEND CMAKE_PREFIX_PATH "${ROCM_ROOT}")

# Find packages
if (BUILD_GPU_LANGUAGE STREQUAL "HIP")
  find_package(hipblas REQUIRED)
  set(DEPENDENCIES roc::hipblas)
  # We cannot be sure which compiler reads common header every time
  add_compile_definitions(__USE_HIP__)
else()
  find_package(CUDAToolkit REQUIRED)
  set(DEPENDENCIES CUDA::cublas)
endif()

# Create SHARED or STATIC host library.
set(library_name Test)
add_library(${library_name} SHARED library/library.cpp)
target_include_directories(${library_name} PRIVATE ${include_dirs})
target_include_directories(${library_name} PUBLIC library)
target_link_libraries(${library_name} PRIVATE ${DEPENDENCIES})
set_target_properties(${library_name} PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_source_files_properties(library/library.cpp PROPERTIES LANGUAGE ${BUILD_GPU_LANGUAGE})

# Create a driver executable using ROCm's bundled version of clang.
set(PROJECT_NAME_clang ${PROJECT_NAME}_clang)
add_executable(${PROJECT_NAME_clang} main.cpp)
target_link_libraries(${PROJECT_NAME_clang} PRIVATE ${library_name})
set_target_properties(${PROJECT_NAME_clang} PROPERTIES HIP_ARCHITECTURES FALSE)

# Create a driver executable using the host c++ compiler.
set(PROJECT_NAME_cxx ${PROJECT_NAME}_cxx)
add_executable(${PROJECT_NAME_cxx} main.cpp)
target_link_libraries(${PROJECT_NAME_cxx} PRIVATE ${library_name})
set_target_properties(${PROJECT_NAME_cxx} PROPERTIES LINKER_LANGUAGE CXX)

# Create ctests:
enable_testing()
add_test(NAME ${PROJECT_NAME_clang} COMMAND ${PROJECT_NAME_clang})
add_test(NAME ${PROJECT_NAME_cxx} COMMAND ${PROJECT_NAME_cxx})
