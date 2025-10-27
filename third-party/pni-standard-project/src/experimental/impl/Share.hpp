#pragma once
#include "include/basic/CudaPtr.hpp"
#include "include/experimental/core/Mich.hpp"
#define REGISTER_THREAD_LOCAL_CUDA_PTR(Typename, Funcname, Ptrname)                                                    \
  inline openpni::cuda_sync_ptr<Typename> &Funcname() {                                                                \
    thread_local static openpni::cuda_sync_ptr<Typename> ptr{Ptrname};                                                 \
    return ptr;                                                                                                        \
  }

REGISTER_THREAD_LOCAL_CUDA_PTR(
    openpni::experimental::core::MichStandardEvent, tl_mich_standard_events, "MichStandardEvents(thread_local)")
REGISTER_THREAD_LOCAL_CUDA_PTR(
    std::size_t, tl_lorBatch_indices, "LORBatch_indices(thread_local)")

#undef REGISTER_THREAD_LOCAL_CUDA_PTR
