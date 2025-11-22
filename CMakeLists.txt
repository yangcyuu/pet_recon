cmake_minimum_required(VERSION 4.1)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(BUILD_SHARED_LIBS OFF)

set(THIRD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third-party)
set(CXX_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(CXX_HEADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    if (CMAKE_SYSTEM_NAME MATCHES "Darwin")
        if (CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
            set(LLVM_PATH "/opt/homebrew/opt/llvm")
        elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
            set(LLVM_PATH "/usr/local/opt/llvm")
        endif ()
        link_directories("${LLVM_PATH}/lib")
        link_directories("${LLVM_PATH}/lib/c++")
    endif ()
endif ()

if (CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
    add_compile_options(-march=native)
    add_compile_options(-funroll-loops)
endif ()

project(pet_recon)

enable_language(CXX CUDA)

set(LIBTORCH_CMAKE_PREFIX "/opt/libtorch/share/cmake/Torch")
set(CUDA_INCLUDE_DIR "/opt/cuda/include")
set(CUDA_CCCL_INCLUDE_DIR "/opt/cuda/include/cccl")
set(CUDA_LIBRARY_DIR "/opt/cuda/lib64")

# set(CUDA_NVRTC_SHORTHASH 1)
list(APPEND CMAKE_PREFIX_PATH ${LIBTORCH_CMAKE_PREFIX})

find_package(CUDAToolkit REQUIRED)
find_package(Torch CONFIG REQUIRED)

# add_subdirectory(${THIRD_PARTY_DIR}/drjit)
add_subdirectory(${THIRD_PARTY_DIR}/pni-standard-project)

include_directories(${CXX_HEADER_DIR})


if (DRJIT_ENABLE_CUDA)
    add_definitions(-DENABLE_CUDA)
    message("-- CUDA support enabled")
else ()
    message("-- CUDA support disabled")
endif ()


file(GLOB_RECURSE CXX_SOURCES
        ${CXX_SOURCE_DIR}/*.cpp
)

file(GLOB_RECURSE CXX_HEADERS
        ${CXX_HEADER_DIR}/*.h
        ${CXX_HEADER_DIR}/*.hpp
)

add_executable(pet_recon)

target_link_libraries(pet_recon PRIVATE
        pni
        pni_cu
        ${TORCH_LIBRARIES}
)

target_sources(pet_recon PRIVATE
        ${CXX_SOURCES}
        ${CXX_HEADERS}
)


add_executable(pni_recon
        ${CMAKE_CURRENT_SOURCE_DIR}/third-party/pni-standard-project/manual-test/Recon/overall_osem_simple_gpu.cpp
)


target_link_libraries(pni_recon PRIVATE
        pni
        pni_cu
)
