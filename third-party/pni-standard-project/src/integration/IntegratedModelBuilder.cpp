#include "IntegratedModelBuilder.hpp"

namespace openpni::basic {
IntegratedModelBuilder::IntegratedModelBuilder() noexcept
    : m_impl(std::make_unique<IntegratedModelBuilder_impl>()) {}

IntegratedModelBuilder::~IntegratedModelBuilder() noexcept {}

void IntegratedModelBuilder::operator<<(
    std::shared_ptr<device::DetectorBase> &&__detectorRuntime) noexcept {
  (*m_impl) << std::move(__detectorRuntime);
}

std::unique_ptr<IntegratedModel> IntegratedModelBuilder::toModel() noexcept {
  return m_impl->model();
}
} // namespace openpni::basic
std::vector<unsigned> genCrystalNumPrefixSum(
    const std::vector<std::shared_ptr<openpni::device::DetectorBase>> &detectors) {
  std::vector<unsigned> crystalNumPrefixSum(detectors.size() + 1, 0);
  unsigned count = 0;
  for (size_t i = 0; i < detectors.size(); ++i) {
    auto detector = detectors[i];
    auto unchanged = detector->detectorUnchangable();
    crystalNumPrefixSum[i] = count;
    count += static_cast<unsigned>(unchanged.geometry.blockNumU) * static_cast<unsigned>(unchanged.geometry.blockNumV) *
             static_cast<unsigned>(unchanged.geometry.crystalNumU) * static_cast<unsigned>(unchanged.geometry.crystalNumV);
  }
  crystalNumPrefixSum[detectors.size()] = count;
  return crystalNumPrefixSum;
}
using CryGeo = openpni::basic::CrystalGeometry;
std::vector<CryGeo> genCrystalPositions(
    const std::vector<std::shared_ptr<openpni::device::DetectorBase>> &detectors) {
  std::vector<CryGeo> crystalPositions;
  for (const auto &detector : detectors) {
    auto unchanged = detector->detectorUnchangable();
    const auto &geometry = unchanged.geometry;
    for (uint8_t u = 0; u < geometry.blockNumU; ++u) {
      for (uint8_t v = 0; v < geometry.blockNumV; ++v) {
        for (uint16_t cu = 0; cu < geometry.crystalNumU; ++cu) {
          for (uint16_t cv = 0; cv < geometry.crystalNumV; ++cv) {
            crystalPositions.push_back(openpni::basic::calculateCrystalGeometry(detector->detectorChangable().coordinate, geometry, cu, cv));
          }
        }
      }
    }
  }
  return crystalPositions;
}
openpni::process::AcquisitionInfo genAcquisitionParams(
    const std::vector<std::shared_ptr<openpni::device::DetectorBase>> &detectors) {
  openpni::process::AcquisitionInfo acquisitionParams;
  if (detectors.empty())
    return acquisitionParams;

  acquisitionParams.totalChannelNum = static_cast<uint16_t>(detectors.size());
  acquisitionParams.storageUnitSize = detectors.front()->detectorUnchangable().maxUDPPacketSize;
  for (const auto detectorIndex : std::ranges::views::iota(0u, detectors.size())) {
    auto detector = detectors[detectorIndex];
    auto unchanged = detector->detectorUnchangable();
    auto &changed = detector->detectorChangable();
    acquisitionParams.storageUnitSize = std::max(acquisitionParams.storageUnitSize, unchanged.maxUDPPacketSize);

    acquisitionParams.channelSettings.push_back(
        {changed.ipSource, changed.portSource, changed.ipDestination, changed.portDestination, static_cast<uint16_t>(detectorIndex),
         [minPktSize = unchanged.minUDPPacketSize, maxPktSize = unchanged.maxUDPPacketSize](
             uint8_t *__udpDatagram, uint16_t __udpLength, uint32_t __ipSource, uint16_t __portSource) noexcept -> bool {
           if (__udpLength < minPktSize)
             return false;
           if (__udpLength > maxPktSize)
             return false;
           return true;
         }});
  }

  return acquisitionParams;
}

namespace openpni::basic {
std::unique_ptr<IntegratedModel> IntegratedModelBuilder_impl::model() noexcept {
  auto model_impl = std::make_unique<IntegratedModel_impl>();
  model_impl->m_detectors = m_detectors;
  model_impl->m_crystalNumPrefixSum = genCrystalNumPrefixSum(m_detectors);
  model_impl->m_crystalGeometry = genCrystalPositions(m_detectors);
  model_impl->m_acquisitionParams_unfinished = genAcquisitionParams(m_detectors);

  return std::make_unique<IntegratedModel>(std::move(model_impl));
}
} // namespace openpni::basic
