
#include "include/experimental/node/MichScatter.hpp"

#include "impl/Copy.h"
#include "impl/MichScatterImpl.hpp"
#include "include/Exceptions.hpp"
#include "include/basic/CudaPtr.hpp"
namespace openpni::experimental::node {
MichScatter::MichScatter(
    core::MichDefine mich)
    : MichScatter(std::make_unique<MichScatter_impl>(mich)) {}
MichScatter::~MichScatter() {}
MichScatter::MichScatter(
    std::unique_ptr<MichScatter_impl> impl)
    : m_impl(std::move(impl)) {}
MichScatter::MichScatter(MichScatter &&) noexcept = default;
MichScatter &MichScatter::operator=(
    MichScatter &&) noexcept = default;

void MichScatter::setScatterPointsThreshold(
    double v) {
  m_impl->setScatterPointsThreshold(v);
}
void MichScatter::setTailFittingThreshold(
    double v) {
  m_impl->setTailFittingThreshold(v);
}
void MichScatter::setScatterEnergyWindow(
    core::Vector<double, 3> windows) {
  m_impl->setScatterEnergyWindow(windows);
}
void MichScatter::setScatterEnergyWindow(
    double low, double high, double resolution) {
  m_impl->setScatterEnergyWindow(low, high, resolution);
}
void MichScatter::setScatterEffTableEnergy(
    core::Vector<double, 3> energies) {
  m_impl->setScatterEffTableEnergy(energies);
}
void MichScatter::setScatterEffTableEnergy(
    double low, double high, double interval) {
  m_impl->setScatterEffTableEnergy(low, high, interval);
}
void MichScatter::setMinSectorDifference(
    int v) {
  m_impl->setMinSectorDifference(v);
}
void MichScatter::setTOFParams(
    double timeBinWidth, double timeBinStart, double timeBinEnd, double systemTimeRes_ns) {
  m_impl->setTOFParams(timeBinWidth, timeBinStart, timeBinEnd, systemTimeRes_ns);
}
void MichScatter::bindAttnCoff(
    MichAttn *h_data) {
  m_impl->bindAttnCoff(h_data);
}
void MichScatter::bindNorm(
    MichNormalization *norm) {
  m_impl->bindNorm(norm);
}
void MichScatter::bindHEmissionMap(
    core::Grids<3, float> emap, float const *h_data) {
  m_impl->bindHEmissionMap(emap, h_data);
}
void MichScatter::bindRandom(
    MichRandom *random) {
  m_impl->bindRandom(random);
}
void MichScatter::bindHPromptMich(
    float *h_promptMich) {
  m_impl->bindHPromptMich(h_promptMich);
}
void MichScatter::bindDEmissionMap(
    core::Grids<3, float> emap, float const *d_data) {
  m_impl->bindDEmissionMap(emap, d_data);
}
void MichScatter::bindDPromptMich(
    float *d_promptMich) {
  m_impl->bindDPromptMich(d_promptMich);
}
float const *MichScatter::getHScatterFactorsBatch(
    std::span<core::MichStandardEvent const> events) {
  return m_impl->getHScatterFactorsBatch(events);
}
float const *MichScatter::getDScatterFactorsBatch(
    std::span<core::MichStandardEvent const> events) {
  return m_impl->getDScatterFactorsBatch(events);
}

std::unique_ptr<float[]> MichScatter::dumpScatterMich() {
  return m_impl->dumpHScatterMich();
}
// void MichScatter::bindHListmode(
//     std::span<basic::Listmode_t const> listmodes) {
//   m_impl->bindHListmode(listmodes);
// }
void MichScatter::bindDListmode(
    std::span<basic::Listmode_t const> listmodes) {
  m_impl->bindDListmode(listmodes);
}
void MichScatter::setScatterPointGrid(
    core::Grids<3> grid) {
  m_impl->setScatterPointGrid(grid);
}

} // namespace openpni::experimental::node
