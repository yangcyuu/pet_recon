#pragma once
#if !PNI_STANDARD_CONFIG_DISABLE_CUDA
#include <array>
#include <cstdint>
#include <cuda_runtime.h>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "../Exceptions.hpp"

namespace openpni::basic::cuda_ptr {
inline cudaStream_t default_stream() {
  thread_local static cudaStream_t stream = cudaStreamPerThread;
  return stream;
}
inline void cuda_throw(
    cudaError_t __err, std::string __msg) {
  switch (__err) {
  case cudaSuccess:
    break;
  case cudaErrorInvalidValue:
    throw exceptions::cuda_error_invalid_value(__msg);
    break;
  case cudaErrorNotSupported:
    throw exceptions::cuda_error_not_support(__msg);
    break;
  case cudaErrorMemoryAllocation:
    throw exceptions::cuda_error_memory_allocation(__msg);
    break;
  default:
    throw exceptions::cuda_error_unknown(__msg + " (CUDA SAY: " + cudaGetErrorString(__err) + ")");
    break;
  }
}
inline void cuda_alloc_throw(
    std::string __msg) {
  cuda_throw(cudaGetLastError(), __msg);
}
} // namespace openpni::basic::cuda_ptr
namespace openpni {
inline int cuda_get_device_index_exept() noexcept(
    false) {
  int deviceIndex;
  cudaError_t err = cudaGetDevice(&deviceIndex);
  basic::cuda_ptr::cuda_throw(err, "Failed to get current CUDA device index");
  return deviceIndex;
}
} // namespace openpni

namespace openpni::basic::cuda_ptr {
struct cuda_device_index_guard {
  int m_oldIndex;
  cuda_device_index_guard() { cudaGetDevice(&m_oldIndex); }
  ~cuda_device_index_guard() { cudaSetDevice(m_oldIndex); }
};

enum class CudaPtrType : uint8_t { sync, async };
template <CudaPtrType T>
struct cuda_ptr_allocator {};
template <>
struct cuda_ptr_allocator<CudaPtrType::sync> {
  cuda_ptr_allocator(
      cudaStream_t __stream) noexcept
      : m_stream(__stream) {}
  cuda_ptr_allocator() noexcept
      : m_stream(default_stream()) {}
  template <typename U>
  U *allocate(
      std::size_t __n, std::string __name) const {
    U *ptr;
    cuda_throw(cudaMalloc((void **)&ptr, __n * sizeof(U)),
               "trying to allocate " + std::to_string(__n * sizeof(U)) + " bytes of type " + typeid(U).name());
    CUDA_ALLOC_DEBUG(__n * sizeof(U), __name)
    return ptr;
  }
  template <typename U>
  void deallocate(
      U *__p, std::size_t __n, std::string __name) const {
    if (__p != nullptr) {
      cudaStreamSynchronize(m_stream);
      cuda_throw(cudaFree(__p),
                 "trying to deallocate " + std::to_string(__n * sizeof(U)) + " bytes of type " + typeid(U).name());
      CUDA_FREE_DEBUG(__n * sizeof(U), __name)
    }
  }
  template <typename U>
  void copy_from_host_to_device(
      U *__d_dst, std::span<const U> __h_src) const {
    cuda_throw(cudaMemcpy(__d_dst, __h_src.data(), __h_src.size() * sizeof(U), cudaMemcpyHostToDevice),
               "Failed to copy memory to device");
  }
  template <typename U>
  void copy_from_device_to_host(
      U *__h_dst, std::span<const U> __d_src) const {
    cuda_throw(cudaMemcpy(__h_dst, __d_src.data(), __d_src.size() * sizeof(U), cudaMemcpyDeviceToHost),
               "Failed to copy memory to host");
  }
  template <typename U>
  void copy_from_device_to_device(
      U *__d_dst, std::span<const U> __d_src) const {
    cuda_throw(cudaMemcpy(__d_dst, __d_src.data(), __d_src.size() * sizeof(U), cudaMemcpyDeviceToDevice),
               "Failed to copy memory to device");
  }
  template <typename U>
  void memset(
      int __value, std::span<U> __d_ptr) const {
    cuda_throw(cudaMemset(__d_ptr.data(), __value, __d_ptr.size() * sizeof(U)), "Failed to set memory on device");
  }
  auto stream() const { return m_stream; }

private:
  cudaStream_t m_stream;
};
template <>
struct cuda_ptr_allocator<CudaPtrType::async> {
  cuda_ptr_allocator(
      cudaStream_t __stream) noexcept
      : m_stream(__stream) {}
  cuda_ptr_allocator() noexcept
      : m_stream(default_stream()) {}
  template <typename U>
  U *allocate(
      std::size_t __n, std::string __name) const {
    U *ptr;
    cuda_throw(cudaMallocAsync((void **)&ptr, __n * sizeof(U), m_stream),
               "trying to allocate " + std::to_string(__n * sizeof(U)) + " bytes of type " + typeid(U).name());
    CUDA_ALLOC_DEBUG(__n * sizeof(U), __name)
    return ptr;
  }
  template <typename U>
  void deallocate(
      U *__p, std::size_t __n, std::string __name) const {
    if (__p != nullptr) {
      cudaStreamSynchronize(m_stream);
      cuda_throw(cudaFreeAsync(__p, m_stream),
                 "trying to deallocate " + std::to_string(__n * sizeof(U)) + " bytes of type " + typeid(U).name());
      CUDA_FREE_DEBUG(__n * sizeof(U), __name)
    }
  }
  template <typename U>
  void copy_from_host_to_device(
      U *__d_dst, std::span<const U> __h_src) const {
    cuda_throw(cudaMemcpyAsync(__d_dst, __h_src.data(), __h_src.size() * sizeof(U), cudaMemcpyHostToDevice, m_stream),
               "Failed to copy memory to device");
  }
  template <typename U>
  void copy_from_device_to_device(
      U *__d_dst, std::span<const U> __d_src) const {
    cuda_throw(cudaMemcpyAsync(__d_dst, __d_src.data(), __d_src.size() * sizeof(U), cudaMemcpyDeviceToDevice, m_stream),
               "Failed to copy memory to device");
  }
  template <typename U>
  void copy_from_device_to_host(
      U *__h_dst, std::span<const U> __d_src) const {
    cuda_throw(cudaMemcpyAsync(__h_dst, __d_src.data(), __d_src.size() * sizeof(U), cudaMemcpyDeviceToHost, m_stream),
               "Failed to copy memory to host");
  }
  template <typename U>
  void memset(
      int __value, std::span<U> __d_ptr) const {
    cuda_throw(cudaMemsetAsync(__d_ptr.data(), __value, __d_ptr.size() * sizeof(U), m_stream),
               "Failed to set memory on device");
  }

private:
  cudaStream_t m_stream;
};

template <typename T, CudaPtrType Type>
class cuda_unique_ptr {
private:
  T *m_d_ptr{nullptr};
  std::size_t m_elements{0};
  int m_deviceIndex{0};
  cuda_ptr_allocator<Type> m_allocator;
  std::string m_name;

public:
  cuda_unique_ptr(
      std::string __name = "Unnamed") noexcept
      : m_allocator(default_stream()) {
    m_d_ptr = nullptr;
    m_elements = 0;
    m_deviceIndex = cuda_get_device_index_exept();
    m_name = __name;
  };

  cuda_unique_ptr(
      uint64_t __elements, cudaStream_t __stream, std::string __name = "Unnamed") noexcept
      : cuda_unique_ptr(__name) {
    if (__elements == 0)
      return;
    m_allocator = cuda_ptr_allocator<Type>(__stream);
    m_d_ptr = m_allocator.template allocate<T>(__elements, __name);
    m_name = std::move(__name);
    m_elements = __elements;
  }

  // No copy constructor
  cuda_unique_ptr(const cuda_unique_ptr<T, Type> &__src) noexcept = delete;
  // No copy assignment operator
  cuda_unique_ptr<T, Type> &operator=(const cuda_unique_ptr<T, Type> &__src) noexcept = delete;

  // Move constructor
  cuda_unique_ptr(
      cuda_unique_ptr<T, Type> &&__src) noexcept {
    m_d_ptr = __src.m_d_ptr;
    m_deviceIndex = __src.m_deviceIndex;
    m_elements = __src.m_elements;
    m_allocator = std::move(__src.m_allocator);
    m_name = std::move(__src.m_name);
    __src.m_d_ptr = nullptr;
    __src.m_elements = 0;
  };

  // Move assignment operator
  cuda_unique_ptr<T, Type> &operator=(
      cuda_unique_ptr<T, Type> &&__src) {
    if (this->m_d_ptr != nullptr) {
      cuda_device_index_guard guard;
      cudaSetDevice(this->m_deviceIndex);
      m_allocator.deallocate(this->m_d_ptr, this->m_elements, m_name);
      this->m_d_ptr = nullptr;
    }

    this->m_d_ptr = __src.m_d_ptr;
    this->m_deviceIndex = __src.m_deviceIndex;
    this->m_elements = __src.m_elements;
    this->m_allocator = std::move(__src.m_allocator);
    this->m_name = std::move(__src.m_name);
    __src.m_d_ptr = nullptr;
    __src.m_elements = 0;

    return *this;
  }

  // Destructor
  ~cuda_unique_ptr() {
    cuda_device_index_guard guard;
    cudaSetDevice(m_deviceIndex);
    if (m_d_ptr != nullptr)
      m_allocator.deallocate(m_d_ptr, m_elements, m_name);
    m_d_ptr = nullptr;
  };

public: // Common interface
  void swap(
      cuda_unique_ptr<T, Type> &__other) noexcept {
    std::swap(m_d_ptr, __other.m_d_ptr);
    std::swap(m_deviceIndex, __other.m_deviceIndex);
    std::swap(m_elements, __other.m_elements);
  }
  T *data() const noexcept { return m_d_ptr; };
  T *get() const noexcept { return data(); };
  T *begin() const noexcept { return data(); }
  T *end() const noexcept { return data() + m_elements; }

  operator bool() const noexcept { return m_d_ptr != nullptr; }

  operator T *() const noexcept { return m_d_ptr; }

  int getDeviceIndex() const noexcept { return m_deviceIndex; };

  uint64_t elements() const noexcept { return m_elements; };

  void memset(
      int __value) noexcept {
    if (m_d_ptr == nullptr)
      return;
    m_allocator.memset(__value, span());
  }

  const auto &allocator() const noexcept { return m_allocator; }

  friend void swap(
      cuda_unique_ptr &a, cuda_unique_ptr &b) noexcept {
    a.swap(b);
  }
  auto span() const noexcept { return std::span<T>(m_d_ptr, m_elements); }
  auto cspan() const noexcept { return std::span<const T>(m_d_ptr, m_elements); }
  auto span(
      std::size_t __begin, std::size_t __end) {
    if (__begin >= __end || __end > m_elements)
      throw exceptions::algorithm_unexpected_condition("cuda_unique_ptr: subspan out of range");
    return std::span<T>(m_d_ptr + __begin, __end - __begin);
  }
  auto cspan(
      std::size_t __begin, std::size_t __end) const {
    if (__begin >= __end || __end > m_elements)
      throw exceptions::algorithm_unexpected_condition("cuda_unique_ptr: subspan out of range");
    return std::span<const T>(m_d_ptr + __begin, __end - __begin);
  }
  auto span(
      std::size_t __count) {
    return span(0, __count);
  }
  auto cspan(
      std::size_t __count) const {
    return cspan(0, __count);
  }
  void clear() { *this = {m_name}; }
  void reserve(
      std::size_t __elements) {
    if (__elements <= m_elements)
      return;
    clear();
    m_d_ptr = m_allocator.template allocate<T>(__elements, m_name + "(reserve)");
    m_elements = __elements;
  }

  std::string getName() const { return m_name; }
};

} // namespace openpni::basic::cuda_ptr
namespace openpni::private_common {
template <typename T>
concept StdVector_c = std::is_same_v<std::decay_t<T>, std::vector<typename T::value_type, typename T::allocator_type>>;
template <typename T>
concept StdArray_c = std::is_same_v<std::decay_t<T>, std::array<typename T::value_type, T::size()>>;
template <typename T>
concept StdContinuousContainer_c = StdVector_c<T> || StdArray_c<T>;
template <typename T>
concept StdUniquePtr_c = std::is_same_v<std::remove_cv_t<std::remove_reference_t<T>>,
                                        std::unique_ptr<typename T::element_type, typename T::deleter_type>> ||
                         std::is_same_v<std::remove_cv_t<std::remove_reference_t<T>>,
                                        std::unique_ptr<typename T::element_type[], typename T::deleter_type>>;
template <typename T>
concept StdSharedPtr_c =
    std::is_same_v<std::remove_cv_t<std::remove_reference_t<T>>, std::shared_ptr<typename T::element_type>> ||
    std::is_same_v<std::remove_cv_t<std::remove_reference_t<T>>, std::shared_ptr<typename T::element_type[]>>;
template <typename T>
concept StdSmartPtr_c = StdUniquePtr_c<T> || StdSharedPtr_c<T>;
template <typename T>
concept StdSpan_c =
    std::is_same_v<std::remove_cv_t<std::remove_reference_t<T>>,
                   std::span<const typename T::element_type, T::extent>> ||
    std::is_same_v<std::remove_cv_t<std::remove_reference_t<T>>, std::span<typename T::element_type, T::extent>>;

} // namespace openpni::private_common
namespace openpni {
template <typename T>
using cuda_sync_ptr = basic::cuda_ptr::cuda_unique_ptr<T, basic::cuda_ptr::CudaPtrType::sync>;
template <typename T>
using cuda_async_ptr = basic::cuda_ptr::cuda_unique_ptr<T, basic::cuda_ptr::CudaPtrType::async>;

template <typename T>
inline T *aligned_as(
    void *__origin, int __aligned = 1024) {
  return reinterpret_cast<T *>((reinterpret_cast<std::uintptr_t>(__origin) + (__aligned - 1)) & ~(__aligned - 1));
}

template <typename T>
inline auto make_cuda_sync_ptr(
    std::size_t __num, std::string __name = "Unnamed") {
  return cuda_sync_ptr<T>(__num, basic::cuda_ptr::default_stream(), __name);
}
template <typename T>
inline auto make_cuda_async_ptr(
    std::size_t __num, cudaStream_t __stream) {
  return cuda_async_ptr<T>(__num, __stream);
}
template <typename T>
inline auto make_cuda_sync_ptr_from_dcopy(
    std::span<const T> __d_ptr, std::string __name = "Unnamed") {
  auto result = cuda_sync_ptr<T>(__d_ptr.size(), basic::cuda_ptr::default_stream(), __name);
  result.allocator().copy_from_device_to_device(result.get(), __d_ptr);
  return result;
}
template <typename T>
inline auto make_cuda_async_ptr_from_dcopy(
    std::span<const T> __d_ptr, cudaStream_t __stream) {
  auto result = cuda_async_ptr<T>(__d_ptr.size(), __stream);
  result.allocator().copy_from_device_to_device(result.get(), __d_ptr);
  return result;
}
template <typename T>
inline auto make_cuda_sync_ptr_from_hcopy(
    std::span<const T> __h_ptr, std::string __name = "Unnamed") {
  auto result = cuda_sync_ptr<T>(__h_ptr.size(), basic::cuda_ptr::default_stream(), __name);
  result.allocator().copy_from_host_to_device(result.get(), __h_ptr);
  return result;
}
template <typename T>
inline auto make_cuda_async_ptr_from_hcopy(
    std::span<const T> __h_ptr, cudaStream_t __stream) {
  auto result = cuda_async_ptr<T>(__h_ptr.size(), __stream);
  result.allocator().copy_from_host_to_device(result.get(), __h_ptr);
  return result;
}
template <typename T>
  requires std::is_trivially_copyable_v<T> && (!private_common::StdSpan_c<T>)
inline auto make_cuda_sync_ptr_from_hcopy(
    T &&__h_value, std::string __name = "Unnamed") {
  auto result = cuda_sync_ptr<T>(1, basic::cuda_ptr::default_stream(), __name);
  result.allocator().copy_from_host_to_device(result.get(), std::span<const T>(&__h_value, 1));
  return result;
}
template <typename T>
  requires std::is_trivially_copyable_v<T> && (!private_common::StdSpan_c<T>)
inline auto make_cuda_async_ptr_from_hcopy(
    T &&__h_value, cudaStream_t __stream) {
  auto result = cuda_async_ptr<T>(1, __stream);
  result.allocator().copy_from_host_to_device(result.get(), std::span<const T>(&__h_value, 1));
  return result;
}
template <private_common::StdContinuousContainer_c HostArray>
inline auto make_cuda_sync_ptr_from_hcopy(
    const HostArray &__container, std::string __name = "Unnamed") {
  return make_cuda_sync_ptr_from_hcopy(
      std::span<const typename HostArray::value_type>(&__container[0], __container.size()), __name);
}
template <private_common::StdContinuousContainer_c HostArray>
inline auto make_cuda_async_ptr_from_hcopy(
    const HostArray &__container, cudaStream_t __stream) {
  return make_cuda_async_ptr_from_hcopy(
      std::span<const typename HostArray::value_type>(&__container[0], __container.size()), __stream);
}
template <private_common::StdContinuousContainer_c HostArray>
inline auto make_cuda_sync_ptr_from_hcopy(
    HostArray &&__container, std::string __name = "Unnamed") {
  return make_cuda_sync_ptr_from_hcopy(
      std::span<const typename HostArray::value_type>(&__container[0], __container.size()), __name);
}
template <private_common::StdContinuousContainer_c HostArray>
inline auto make_cuda_async_ptr_from_hcopy(
    HostArray &&__container, cudaStream_t __stream) {
  return make_cuda_async_ptr_from_hcopy(
      std::span<const typename HostArray::value_type>(&__container[0], __container.size()), __stream);
}
template <typename T>
inline auto make_vector_from_cuda_sync_ptr(
    const cuda_sync_ptr<T> &__d_ptr, std::span<const T> __span) {
  if (!__d_ptr)
    return std::vector<T>();
  std::vector<T> result(__span.size());
  __d_ptr.allocator().copy_from_device_to_host(&result[0], __span);
  return result;
}
template <typename T>
inline auto make_vector_from_cuda_async_ptr(
    const cuda_async_ptr<T> &__d_ptr, std::span<const T> __span) {
  if (!__d_ptr)
    return std::vector<T>();
  std::vector<T> result(__span.size());
  __d_ptr.allocator().copy_from_device_to_host(&result[0], __span);
  return result;
}
template <typename T>
inline auto make_vector_from_cuda_sync_ptr(
    const cuda_sync_ptr<T> &__d_ptr) {
  return make_vector_from_cuda_sync_ptr(__d_ptr, std::span<const T>(__d_ptr.begin(), __d_ptr.end()));
}
template <typename T>
inline auto make_vector_from_cuda_async_ptr(
    const cuda_async_ptr<T> &__d_ptr) {
  return make_vector_from_cuda_async_ptr(__d_ptr, std::span<const T>(__d_ptr.begin(), __d_ptr.end()));
}

} // namespace openpni

#endif
