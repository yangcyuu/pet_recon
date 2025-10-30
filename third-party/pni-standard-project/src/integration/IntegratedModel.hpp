#pragma once
#include <ranges>

#include "include/basic/CudaPtr.hpp"
#include "include/basic/Model.hpp"
#include "include/detector/Detectors.hpp"
#include "include/process/Acquisition.hpp"
namespace openpni::basic {
struct CrystalIndexMapper_CPU_impl {
  std::vector<unsigned> crystalNumPrefixSum;
};
#ifndef PNI_STANDARD_CONFIG_DISABLE_CUDA
struct CrystalIndexMapper_CUDA_impl {
  cuda_sync_ptr<unsigned> crystalNumPrefixSum;
};
#endif // !PNI_STANDARD_CONFIG_DISABLE_CUDA

class IntegratedModel_impl {
public:
  friend class IntegratedModelBuilder_impl;
  IntegratedModel_impl() {}
  ~IntegratedModel_impl() {}

public:
  const std::vector<unsigned> &crystalNumPrefixSum() const noexcept { return m_crystalNumPrefixSum; }
  std::vector<device::DetectorBase *> detectorRuntimes() const noexcept {
    auto range = m_detectors | std::ranges::views::transform([](const auto &detector) noexcept { return detector.get(); });
    return std::vector(range.begin(), range.end());
  }
  unsigned detectorNum() const noexcept { return static_cast<unsigned>(m_detectors.size()); }
  const std::vector<basic::CrystalGeometry> &crystalGeometry() const noexcept { return m_crystalGeometry; }
  process::AcquisitionInfo acquisitionParams(
      uint64_t __maxBufferSize, uint16_t __timeSwitchBuffer_ms) const noexcept {
    auto result = m_acquisitionParams_unfinished;
    result.maxBufferSize = __maxBufferSize;
    result.timeSwitchBuffer_ms = __timeSwitchBuffer_ms;
    return result;
  }

private:
  std::vector<std::shared_ptr<device::DetectorBase>> m_detectors;
  std::vector<unsigned> m_crystalNumPrefixSum;
  std::vector<basic::CrystalGeometry> m_crystalGeometry;
  process::AcquisitionInfo m_acquisitionParams_unfinished;
};

} // namespace openpni::basic
