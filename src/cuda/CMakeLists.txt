cmake_minimum_required(VERSION 3.26)
project(cuda CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(cuda gemm.cu)

set_target_properties(cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
