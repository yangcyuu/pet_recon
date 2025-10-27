#pragma once
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "../Exceptions.hpp"
#include "../PnI-Config.hpp"
#if !PNI_STANDARD_CONFIG_DISABLE_CUDA
#include <cuda_runtime.h>

namespace openpni::basic::cuda {
template <class T>
class cuda_pinned_unique_ptr {
  static_assert(!std::is_same<T, void>::value, "cuda_pinned_unique_ptr does not support void type");

public:
  cuda_pinned_unique_ptr() noexcept
      : m_ptr(nullptr)
      , m_elementNum(0) {};
  // no copy constructor
  cuda_pinned_unique_ptr(const cuda_pinned_unique_ptr &) = delete;
  // no copy assignment operator
  cuda_pinned_unique_ptr &operator=(const cuda_pinned_unique_ptr &) = delete;

  // move constructor
  cuda_pinned_unique_ptr(
      cuda_pinned_unique_ptr &&other) noexcept {
    m_ptr = other.m_ptr;
    m_elementNum = other.m_elementNum;
    other.m_ptr = nullptr;
    other.m_elementNum = 0;
  }
  // move assignment operator
  cuda_pinned_unique_ptr &operator=(
      cuda_pinned_unique_ptr &&other) noexcept {
    if (this != &other) {
      if (m_ptr) {
        cudaFreeHost(m_ptr);
      }
      m_ptr = other.m_ptr;
      m_elementNum = other.m_elementNum;
      other.m_ptr = nullptr;
      other.m_elementNum = 0;
    }
    return *this;
  }

  cuda_pinned_unique_ptr(
      T *ptr, uint64_t elementNum) noexcept
      : m_ptr(ptr)
      , m_elementNum(elementNum) {
    // if (m_ptr == nullptr || m_elementNum == 0)
    // {
    //     throw exceptions::cuda_error_memory_allocation("Invalid pointer or element
    //     number");
    // }
  }

  ~cuda_pinned_unique_ptr() {
    if (m_ptr) {
      cudaFreeHost(m_ptr);
    }
  }

public:
  // Access the pointer
  T *get() const noexcept { return m_ptr; }

  // Get the size of the allocated memory
  uint64_t getElementNum() const noexcept { return m_elementNum; }

  // Access operator
  T &operator[](
      uint64_t index) {
    return m_ptr[index];
  }

  // Access operator with bounds checking
  T at(
      uint64_t index) {
    if (index >= m_elementNum) {
      throw std::out_of_range("Index out of range");
    }
    return m_ptr[index];
  }

  void swap(
      cuda_pinned_unique_ptr &other) noexcept {
    std::swap(m_ptr, other.m_ptr);
    std::swap(m_elementNum, other.m_elementNum);
  }

  explicit operator bool() const noexcept { return m_ptr != nullptr; }

  operator T *() const noexcept { return m_ptr; }

  T *begin() noexcept { return m_ptr; }

  T *end() noexcept { return m_ptr + m_elementNum; }

private:
  T *m_ptr;              // Pointer to the pinned memory
  uint64_t m_elementNum; // Size of the allocated memory
};

template <class T>
inline cuda_pinned_unique_ptr<T> make_cuda_pinned_unique_ptr(uint64_t __elementNum);
} // namespace openpni::basic::cuda

namespace openpni::basic::cuda {
template <class T>
inline cuda_pinned_unique_ptr<T> make_cuda_pinned_unique_ptr(
    uint64_t __elementNum) {
  T *d_ptr;
  if (__elementNum > 0) {
    cudaError_t err = cudaMallocHost((void **)&d_ptr, __elementNum * sizeof(T));
    if (err != cudaSuccess) {
      throw exceptions::cuda_error_memory_allocation("cudaMallocHost failed: " + std::string(cudaGetErrorString(err)));
    }
  }
  return cuda_pinned_unique_ptr<T>(d_ptr, __elementNum);
}
} // namespace openpni::basic::cuda

namespace openpni::basic::cuda {
template <class T>
class cuda_unique_ptr {
  static_assert(!std::is_same<T, void>::value, "cuda_unique_ptr does not support void type");

public:
  cuda_unique_ptr() noexcept {
    m_ptr = nullptr;
    m_elementNum = 0;
    cudaGetDevice(&m_deviceId);
  };

  cuda_unique_ptr(const cuda_unique_ptr<T> &__src) noexcept = delete;

  cuda_unique_ptr(
      cuda_unique_ptr<T> &&__src) noexcept {
    m_ptr = __src.m_ptr;
    m_deviceId = __src.m_deviceId;
    m_elementNum = __src.m_elementNum;
    __src.m_ptr = nullptr;
    __src.m_elementNum = 0;
    __src.m_deviceId = 0; // Reset the source pointer
  };

  cuda_unique_ptr(
      T *__ptr, const uint64_t __elements, const int __deviceId) noexcept
      : cuda_unique_ptr() {
    m_ptr = __ptr;
    m_elementNum = __elements;
    m_deviceId = __deviceId;
  };

  ~cuda_unique_ptr() noexcept {
    int tempIndex;
    cudaGetDevice(&tempIndex);
    cudaSetDevice(m_deviceId);
    cudaFree(m_ptr);
    m_ptr = nullptr;
    m_elementNum = 0;
    m_deviceId = 0; // Reset the pointer
    cudaSetDevice(tempIndex);
  };

public:
  cuda_unique_ptr<T> &operator=(cuda_unique_ptr<T> &__other) noexcept = delete;

  cuda_unique_ptr<T> &operator=(
      cuda_unique_ptr<T> &&__right) noexcept {
    if (this != &__right) {
      int tempIndex;
      cudaGetDevice(&tempIndex);
      cudaSetDevice(this->m_deviceId);
      cudaFree(this->m_ptr);
      this->m_ptr = __right.m_ptr;
      this->m_deviceId = __right.m_deviceId;
      this->m_elementNum = __right.m_elementNum;
      __right.m_elementNum = 0;
      __right.m_deviceId = 0; // Reset the source pointer
      __right.m_ptr = nullptr;
      cudaSetDevice(tempIndex);
    }
    return *this;
  };

  void swap(
      cuda_unique_ptr<T> &__other) noexcept {
    std::swap(m_ptr, __other.m_ptr);
    std::swap(m_deviceId, __other.m_deviceId);
    std::swap(m_elementNum, __other.m_elementNum);
  }
  T *get() const noexcept { return m_ptr; };
  T at(
      const uint64_t __index) const noexcept {
    T result;
    if (__index >= m_elementNum) {
      throw std::out_of_range("Index out of range");
    }
    cudaMemcpy(&result, m_ptr + __index, sizeof(T), cudaMemcpyDeviceToHost);
    return result;
  };

  explicit operator bool() const noexcept { return m_ptr != nullptr; }
  operator T *() const noexcept { return m_ptr; }

  int getDeviceIndex() const noexcept { return m_deviceId; };
  uint64_t getElementNum() const noexcept { return m_elementNum; };

private:
  T *m_ptr;
  int m_deviceId;
  uint64_t m_elementNum{0};
};

template <typename T>
cuda_unique_ptr<T> make_cuda_unique_ptr(const uint64_t __elementNum = 1);

} // namespace openpni::basic::cuda

namespace openpni::basic::cuda {
template <typename T>
cuda_unique_ptr<T> make_cuda_unique_ptr(
    const uint64_t __elementNum) {
  T *d_ptr;
  int deviceId;
  cudaGetDevice(&deviceId);
  cudaError_t err = cudaMalloc((void **)&d_ptr, __elementNum * sizeof(T));

  switch (err) {
  case cudaSuccess:
    break;
  case cudaErrorInvalidValue:
    throw exceptions::cuda_error_invalid_value();
    break;
  case cudaErrorNotSupported:
    throw exceptions::cuda_error_not_support();
    break;
  case cudaErrorMemoryAllocation:
    throw exceptions::cuda_error_memory_allocation();
    break;
  default:
    throw exceptions::cuda_error_unknown();
    break;
  }

  return cuda_unique_ptr<T>(d_ptr, __elementNum, deviceId);
}
} // namespace openpni::basic::cuda

#endif
