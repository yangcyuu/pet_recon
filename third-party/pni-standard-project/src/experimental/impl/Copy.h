#pragma once
#include <algorithm>

#include "include/basic/CudaPtr.hpp"
template <typename T>
inline std::unique_ptr<T[]> host_deep_copy(
    T const *__begin, std::size_t __count) {
  auto result = std::make_unique_for_overwrite<T[]>(__count);
  std::copy_n(__begin, __count, result.get());
  return result;
}
template <typename T>
inline openpni::cuda_sync_ptr<T> device_deep_copy(
    const openpni::cuda_sync_ptr<T> &__src) {
  return openpni::make_cuda_sync_ptr_from_dcopy(__src.cspan(), "device_deep_copy");
}
template <typename T>
inline std::unique_ptr<T[]> device_deep_copy_to_host(
    const openpni::cuda_sync_ptr<T> &__src) {
  auto result = std::make_unique_for_overwrite<T[]>(__src.elements());
  __src.allocator().copy_from_device_to_host(result.get(), __src.cspan());
  return result;
}
template <typename T>
inline std::unique_ptr<T[]> device_deep_copy_to_host(
    std::span<T const> const __d_span) {
  auto result = std::make_unique_for_overwrite<T[]>(__d_span.size());
  cudaMemcpy(result.get(), __d_span.data(), sizeof(T) * __d_span.size(), cudaMemcpyDeviceToHost);
  return result;
}
template <typename T>
inline openpni::cuda_sync_ptr<T> host_deep_copy_to_device(
    T const *__begin, std::size_t __count) {
  return openpni::make_cuda_sync_ptr_from_hcopy(std::span<T const>(__begin, __count), "host_deep_copy_to_device");
}
