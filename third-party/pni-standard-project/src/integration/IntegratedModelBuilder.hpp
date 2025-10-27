#pragma once
#include "IntegratedModel.hpp"
#include "include/detector/Detectors.hpp"
namespace openpni::basic {
class IntegratedModelBuilder_impl {
public:
  IntegratedModelBuilder_impl() = default;
  ~IntegratedModelBuilder_impl() = default;
  void operator<<(
      std::shared_ptr<device::DetectorBase> &&__detectorRuntime) noexcept {
    m_detectors.push_back(std::move(__detectorRuntime));
  }
  std::unique_ptr<IntegratedModel> model() noexcept;

private:
  std::vector<std::shared_ptr<device::DetectorBase>> m_detectors;
};
} // namespace openpni::basic