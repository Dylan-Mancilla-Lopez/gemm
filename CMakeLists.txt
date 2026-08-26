cmake_minimum_required(VERSION 3.20)
project(gemm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")

option(GEMM_SANITIZE "Build with ASan + UBSan" OFF)
# Kaby Lake R: AVX2 + FMA, sin AVX-512. Todas las versiones se compilan con las mismas flags.
add_compile_options(-Wall -Wextra -mavx2 -mfma)
if(GEMM_SANITIZE)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
endif()

find_package(OpenMP REQUIRED)
add_library(gemm src/simple.cpp src/simple_o3.cpp src/simd.cpp)
set_source_files_properties(src/simple_o3.cpp PROPERTIES COMPILE_OPTIONS "-O3")
target_include_directories(gemm PUBLIC include)
target_link_libraries(gemm PUBLIC OpenMP::OpenMP_CXX)

# OpenBLAS: referencia numérica en los tests y rival en los benchmarks.
find_path(OPENBLAS_INCLUDE cblas.h PATHS /usr/include/openblas NO_DEFAULT_PATH)
find_path(OPENBLAS_INCLUDE cblas.h)
find_library(OPENBLAS_LIB openblas REQUIRED)

enable_testing()
find_package(GTest REQUIRED)
add_executable(gemm_tests tests/test_gemm.cpp)
target_include_directories(gemm_tests PRIVATE ${OPENBLAS_INCLUDE})
target_link_libraries(gemm_tests PRIVATE gemm GTest::gtest_main ${OPENBLAS_LIB})
include(GoogleTest)
gtest_discover_tests(gemm_tests)

add_executable(gemm_bench bench/gemm_bench.cpp)
target_include_directories(gemm_bench PRIVATE ${OPENBLAS_INCLUDE})
target_link_libraries(gemm_bench PRIVATE gemm ${OPENBLAS_LIB})

add_executable(peak bench/peak.cpp)
target_link_libraries(peak PRIVATE OpenMP::OpenMP_CXX)
