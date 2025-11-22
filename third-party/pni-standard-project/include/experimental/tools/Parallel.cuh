#pragma once
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

#include "../../basic/CudaPtr.hpp"
#include "../core/Span.hpp"
namespace openpni::experimental::tools {

template <typename Func>
inline void parallel_for_each_CUDA(
    // For_each from 0 to max-1
    std::size_t __max, Func __func, cudaStream_t __stream) {
  if (__max == 0)
    return;
  thrust::for_each(thrust::cuda::par.on(__stream), thrust::counting_iterator<size_t>(0),
                   thrust::counting_iterator<size_t>(__max), __func);
  // #if PNI_STANDARD_CONFIG_ENABLE_DEBUG
  // basic::cuda_ptr::cuda_throw(cudaStreamSynchronize(__stream), "Wating for parallel_for_each_CUDA to complete");
  // #endif
}
template <typename Func>
inline void parallel_for_each_CUDA(
    // For_each from 0 to max-1 (default stream version)
    std::size_t __max, Func __func) {
  parallel_for_each_CUDA(__max, __func, basic::cuda_ptr::default_stream());
}
template <typename Func>
inline void parallel_for_each_CUDA(
    // For_each from begin to end-1
    std::size_t __begin, std::size_t __end, Func __func, cudaStream_t __stream) {
  if (__begin >= __end)
    return;
  parallel_for_each_CUDA(__end - __begin, [=] __device__(std::size_t index) { __func(index + __begin); }, __stream);
}
template <typename Func>
inline void parallel_for_each_CUDA(
    // For_each from begin to end-1 (default stream version)
    std::size_t __begin, std::size_t __end, Func __func) {
  parallel_for_each_CUDA(__begin, __end, __func, basic::cuda_ptr::default_stream());
}
template <int N, typename Func>
inline void parallel_for_each_CUDA(
    // For_each for each index in MDSpan
    core::MDSpan<N> span, Func &&func, cudaStream_t __stream) {
  parallel_for_each_CUDA(span.totalSize(), [=] __device__(std::size_t index) { func(span.toIndex(index)); }, __stream);
}
template <int N, typename Func>
inline void parallel_for_each_CUDA(
    // For_each for each index in MDSpan (default stream version)
    core::MDSpan<N> span, Func func) {
  parallel_for_each_CUDA(span, func, basic::cuda_ptr::default_stream());
}
template <int N, typename Func>
inline void parallel_for_each_CUDA(
    // For_each for each index in MDBeginEndSpan
    core::MDBeginEndSpan<N> span, Func func, cudaStream_t __stream) {
  parallel_for_each_CUDA(span.totalSize(), [=] __device__(std::size_t index) { func(span.toIndex(index)); }, __stream);
}
template <int N, typename Func>
inline void parallel_for_each_CUDA(
    // For_each for each index in MDBeginEndSpan (default stream version)
    core::MDBeginEndSpan<N> span, Func func) {
  parallel_for_each_CUDA(span, func, basic::cuda_ptr::default_stream());
}
} // namespace openpni::experimental::tools