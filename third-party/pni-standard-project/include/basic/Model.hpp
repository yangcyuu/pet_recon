#pragma once
#include <memory>
#include <vector>

#include "../math/Geometry.hpp"
#include "../process/Acquisition.hpp"
namespace openpni::device {
class DetectorBase;
} // namespace openpni::device

namespace openpni::basic {
class IntegratedModel_impl;
class IntegratedModel {
public:
  IntegratedModel(std::unique_ptr<IntegratedModel_impl> &&impl) noexcept;

public:
  /**
   * * @brief
   * 每个探测器的晶体数量的前缀和，长度为探测器数量+1，最后一个元素为所有探测器的晶体数量之和
   */
  std::vector<device::DetectorBase *> detectorRuntimes() const noexcept;
  const std::vector<unsigned> &crystalNumPrefixSum() const noexcept;
  unsigned crystalNum() const noexcept;
  unsigned detectorNum() const noexcept;
  const std::vector<basic::CrystalGeometry> &crystalGeometry() const noexcept;
  process::AcquisitionInfo acquisitionParams(uint64_t __maxBufferSize, uint16_t __timeSwitchBuffer_ms) const noexcept;

private:
  std::unique_ptr<IntegratedModel_impl> m_impl;
};

class IntegratedModelBuilder_impl;
class IntegratedModelBuilder {
public:
  IntegratedModelBuilder() noexcept;
  ~IntegratedModelBuilder() noexcept;

public:
  void operator<<(std::shared_ptr<device::DetectorBase> &&__detectorRuntime) noexcept;

public:
  std::unique_ptr<IntegratedModel> toModel() noexcept;

private:
  std::unique_ptr<IntegratedModelBuilder_impl> m_impl;
};

} // namespace openpni::basic
