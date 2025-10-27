// This file is auto-generated. Do not edit manually.

#ifndef _PNI_STANDARD_PROJECT_CONFIG_HPP_
#define _PNI_STANDARD_PROJECT_CONFIG_HPP_
#define PNI_STANDARD_DPDK_MBUFS (1024 * 1024 * 4 - 1)
#define PNI_STANDARD_CONFIG_ENABLE_DPDK 0
#define PNI_STANDARD_CONFIG_ENABLE_DPU 0
#define PNI_STANDARD_CONFIG_DISABLE_CUDA 0
#define PNI_STANDARD_CONFIG_ENABLE_DEBUG 0
#define PNI_STANDARD_CONFIG_ENABLE_DEBUG_CUDA_ALLOC 0
#if defined(__CUDA_RUNTIME_H__)
#ifndef __PNI_CUDA_MACRO__
#define __PNI_CUDA_MACRO__ __host__ __device__
#endif // __PNI_CUDA_MACRO__
#ifndef __PNI_CUDA_HOST_ONLY__
#define __PNI_CUDA_HOST_ONLY__ __host__
#endif // __PNI_CUDA_HOST_ONLY__
#ifndef __PNI_CUDA_DEVICE_ONLY__
#define __PNI_CUDA_DEVICE_ONLY__ __device__
#endif // __PNI_CUDA_DEVICE_ONLY__
#else // defined(__CUDA_RUNTIME_H__)

#ifndef __PNI_CUDA_MACRO__
#define __PNI_CUDA_MACRO__
#endif // __PNI_CUDA_MACRO__
#ifndef __PNI_CUDA_HOST_ONLY__
#define __PNI_CUDA_HOST_ONLY__
#endif // __PNI_CUDA_HOST_ONLY__
#ifndef __PNI_CUDA_DEVICE_ONLY__
#define __PNI_CUDA_DEVICE_ONLY__
#endif // __PNI_CUDA_DEVICE_ONLY__
#endif // defined(__CUDA_RUNTIME_H__)
#if PNI_STANDARD_CONFIG_ENABLE_DEBUG
#include <chrono>
#include <iostream>
inline void PNI_DO_DEBUG(std::string const& msg) {
    thread_local static auto now = std::chrono::system_clock::now();
    int msecond = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - now).count();
    std::cerr << std::to_string(msecond) << "\tPNI_DEBUG: " + msg << std::flush;
}
#define PNI_DEBUG(x)  (PNI_DO_DEBUG(x));
#else
#define PNI_DEBUG(x) 
#endif // PNI_STANDARD_CONFIG_ENABLE_DEBUG

#if PNI_STANDARD_CONFIG_ENABLE_DEBUG && PNI_STANDARD_CONFIG_ENABLE_DEBUG_CUDA_ALLOC
#include <termcolor/termcolor.hpp>
inline void CUDA_ALLOC_DEBUG_SAY(size_t bytes, std::string name, bool isRelease) {
    thread_local static std::size_t total_bytes = 0;
    auto print_bytes = [](std::size_t b) { auto s = std::to_string(b);   for (int i = static_cast<int>(s.size()) - 3; i > 0; i -= 3)s.insert(static_cast<std::size_t>(i), ",");  while(s.size()<16)s=" "+s; return s; };
    if (isRelease) {
        total_bytes -= bytes;
    std::cerr << termcolor::green;
    PNI_DO_DEBUG( "Freed " + print_bytes(bytes) + " bytes CUDA, total: " + print_bytes(total_bytes) + " bytes, name: "+name+".\n");
    std::cerr << termcolor::reset;
    } else {
        total_bytes += bytes;
    std::cerr << termcolor::blue;
    PNI_DO_DEBUG( "Alloc " + print_bytes(bytes) + " bytes CUDA, total: " + print_bytes(total_bytes) + " bytes, name: "+name+".\n");
    std::cerr << termcolor::reset;
    }
}
#define CUDA_ALLOC_DEBUG(x, n)  (CUDA_ALLOC_DEBUG_SAY(x, n, false));
#define CUDA_FREE_DEBUG(x, n)   (CUDA_ALLOC_DEBUG_SAY(x, n, true));
#else
#define CUDA_ALLOC_DEBUG(x, n) 
#define CUDA_FREE_DEBUG(x, n) 
#endif // PNI_STANDARD_CONFIG_ENABLE_DEBUG_CUDA_ALLOC


#endif // _PNI_STANDARD_PROJECT_CONFIG_HPP_
