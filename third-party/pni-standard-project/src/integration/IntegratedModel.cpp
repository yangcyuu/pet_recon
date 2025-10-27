#include "IntegratedModel.hpp"
namespace openpni::basic {
IntegratedModel::IntegratedModel(
    std::unique_ptr<IntegratedModel_impl> &&impl) noexcept
    : m_impl(std::move(impl)) {}
const std::vector<unsigned> &IntegratedModel::crystalNumPrefixSum() const noexcept {
  return m_impl->crystalNumPrefixSum();
}
unsigned IntegratedModel::crystalNum() const noexcept {
  return m_impl->crystalNumPrefixSum().back();
}
std::vector<device::DetectorBase *> IntegratedModel::detectorRuntimes() const noexcept {
  return m_impl->detectorRuntimes();
}

unsigned IntegratedModel::detectorNum() const noexcept {
  return m_impl->detectorNum();
}

const std::vector<basic::CrystalGeometry> &IntegratedModel::crystalGeometry() const noexcept {
  return m_impl->crystalGeometry();
}

process::AcquisitionInfo IntegratedModel::acquisitionParams(
    uint64_t __maxBufferSize, uint16_t __timeSwitchBuffer_ms) const noexcept {
  return m_impl->acquisitionParams(__maxBufferSize, __timeSwitchBuffer_ms);
}
} // namespace openpni::basic
