#include "include/experimental/node/MichAttn.hpp"

#include "impl/MichAttnImpl.hpp"

namespace openpni::experimental::node {
MichAttn::MichAttn(
    core::MichDefine __mich)
    : MichAttn(std::make_unique<MichAttn_impl>(__mich)) {}
MichAttn::~MichAttn() {}
MichAttn::MichAttn(
    std::unique_ptr<MichAttn_impl> impl)
    : m_impl(std::move(impl)) {}

MichAttn::MichAttn(MichAttn &&) noexcept = default;
MichAttn &MichAttn::operator=(MichAttn &&) noexcept = default;

void MichAttn::setMapSize(
    core::Grids<3> map) {
  m_impl->setMapSize(map);
}
void MichAttn::bindHHUMap(
    float const *h_data) {
  m_impl->bindHHUMap(h_data);
}
void MichAttn::bindHAttnMap(
    float const *h_data) {
  m_impl->bindHAttnMap(h_data);
}
std::unique_ptr<float[]> MichAttn::dumpAttnMich() {
  return m_impl->dumpHAttnMich();
}
float const *MichAttn::getHAttnFactorsBatch(
    std::span<core::MichStandardEvent const> events) {
  return m_impl->getHAttnFactorsBatch(events);
}
float const *MichAttn::getHAttnFactorsBatch(
    std::span<std::size_t const> lorIndices) {
  return m_impl->getHAttnFactorsBatch(lorIndices);
}
float const *MichAttn::getDAttnFactorsBatch(
    std::span<core::MichStandardEvent const> events) {
  return m_impl->getDAttnFactorsBatch(events);
}
float const *MichAttn::getDAttnFactorsBatch(
    std::span<std::size_t const> lorIndices) {
  return m_impl->getDAttnFactorsBatch(lorIndices);
}
float const *MichAttn::h_getAttnMap() {
  return m_impl->h_getAttnMap();
}
float const *MichAttn::d_getAttnMap() {
  return m_impl->d_getAttnMap();
}
float const *MichAttn::h_getAttnMich() {
  return m_impl->h_getAttnMich();
}
float const *MichAttn::d_getAttnMich() {
  return m_impl->d_getAttnMich();
}
void MichAttn::setPreferredSource(
    AttnFactorSource source) {
  m_impl->setPreferredSource(source);
}

core::MichDefine MichAttn::mich() const {
  return m_impl->mich();
}
void MichAttn::setFetchMode(
    AttnFactorFetchMode mode) {
  m_impl->setFetchMode(mode);
}

// temp
core::Grids<3> MichAttn::getMapGrids() {
  return m_impl->getMapGrids();
}

} // namespace openpni::experimental::node