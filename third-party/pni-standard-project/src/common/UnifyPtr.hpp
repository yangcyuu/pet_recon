#pragma once
#include "include/basic/CudaPtr.hpp"
namespace openpni::common {
template <typename T>
class UnifyPointer {
public:
  UnifyPointer(
      std::string __name = "Unnamed")
      : mh_outerPtr(nullptr)
      , md_outerPtr(nullptr)
      , mh_innerPtr(nullptr)
      , md_innerPtr({__name}) {}

public:
  void setHostOuterPointer(
      T *__h_ptr, std::size_t __elements) {
    mh_outerPtr = __h_ptr;
    m_elements = __elements;
  }
  void setDeviceOuterPointer(
      T *__d_ptr, std::size_t __elements) {
    md_outerPtr = __d_ptr;
    m_elements = __elements;
  }
  void setHostInnerPointer(
      std::unique_ptr<T[]> __h_ptr, std::size_t __elements) {
    mh_innerPtr = std::move(__h_ptr);
    m_elements = __elements;
  }
  void setDeviceInnerPointer(
      cuda_sync_ptr<T> &&__d_ptr) {
    md_innerPtr = std::move(__d_ptr);
    m_elements = md_innerPtr.elements();
  }
  void resetInnerPointers() {
    mh_innerPtr.reset();
    md_innerPtr.clear();
  }
  void resetInnerHostPointer() { mh_innerPtr.reset(); }
  void resetInnerDevicePointer() { md_innerPtr.clear(); }
  std::size_t elements() const noexcept { return m_elements; }

public:
  std::vector<T> dumpToHostVector() {
    std::vector<T> result(m_elements);
    T *h_ptr = HPtr_nullable();
    if (h_ptr) {
      std::copy(h_ptr, h_ptr + m_elements, result.data());
      return result;
    }
    T *d_ptr = DPtr_nullable();
    if (d_ptr) {
      cudaMemcpyAsync(result.data(), d_ptr, sizeof(T) * m_elements, cudaMemcpyDeviceToHost,
                      basic::cuda_ptr::default_stream());
      return result;
    }
    throw std::runtime_error("UnifyPointer: No valid pointer available for dumpToHostVector");
  }
  cuda_sync_ptr<T> dumpToDevicePointer(
      std::string __name = "Unnamed") {
    auto result = make_cuda_sync_ptr<T>(m_elements, __name);
    T *d_ptr = DPtr_nullable();
    if (d_ptr) {
      cudaMemcpyAsync(result.get(), d_ptr, sizeof(T) * m_elements, cudaMemcpyDeviceToDevice,
                      basic::cuda_ptr::default_stream());
      return result;
    }
    T *h_ptr = HPtr_nullable();
    if (h_ptr) {
      result.allocator().copy_from_host_to_device(result.get(), std::span<const T>(h_ptr, m_elements));
      return result;
    }
    throw std::runtime_error("UnifyPointer: No valid pointer available for dumpToDevicePointer");
  }

public:
  T *HPtr_nullable() const noexcept {
    return mh_outerPtr != nullptr ? mh_outerPtr : (mh_innerPtr ? mh_innerPtr.get() : nullptr);
  }
  T *DPtr_nullable() const noexcept {
    return md_outerPtr != nullptr ? md_outerPtr : (md_innerPtr ? md_innerPtr.get() : nullptr);
  }
  T *HPtr_required() {
    if (mh_outerPtr != nullptr)
      return mh_outerPtr;
    if (mh_innerPtr)
      return mh_innerPtr.get();
    if (md_outerPtr) {
      mh_innerPtr = std::make_unique_for_overwrite<T[]>(m_elements);
      cudaMemcpyAsync(mh_innerPtr.get(), md_outerPtr, sizeof(T) * m_elements, cudaMemcpyDeviceToHost,
                      basic::cuda_ptr::default_stream());
      return mh_innerPtr.get();
    }
    if (md_innerPtr) {
      mh_innerPtr = std::make_unique_for_overwrite<T[]>(m_elements);
      md_innerPtr.allocator().copy_from_device_to_host(mh_innerPtr.get(), md_innerPtr.cspan());
      return mh_innerPtr.get();
    }
    throw std::runtime_error("UnifyPointer: No valid pointer available for HPtr_required");
  }
  T *DPtr_required() {
    if (md_outerPtr != nullptr)
      return md_outerPtr;
    if (md_innerPtr)
      return md_innerPtr.get();
    if (mh_outerPtr) {
      md_innerPtr.reserve(m_elements);
      md_innerPtr.allocator().copy_from_host_to_device(md_innerPtr.get(), std::span<const T>(mh_outerPtr, m_elements));
      return md_innerPtr.get();
    }
    if (mh_innerPtr) {
      md_innerPtr.reserve(m_elements);
      md_innerPtr.allocator().copy_from_host_to_device(md_innerPtr.get(),
                                                       std::span<const T>(mh_innerPtr.get(), m_elements));
      return md_innerPtr.get();
    }
    throw std::runtime_error("UnifyPointer: No valid pointer available for DPtr_required");
  }

private:
private:
  std::size_t m_elements{0};
  // Outer pointer on host and device, the ownership is not belonged to this class
  T *mh_outerPtr;
  T *md_outerPtr;
  // Inner pointer on host and device, the ownership is belonged to this class
  std::unique_ptr<T[]> mh_innerPtr;
  cuda_sync_ptr<T> md_innerPtr;
};
} // namespace openpni::common
