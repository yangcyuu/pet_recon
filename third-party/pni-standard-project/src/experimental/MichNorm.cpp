#include "include/experimental/node/MichNorm.hpp"

#include <variant>

#include "impl/MichNormImpl.hpp"
namespace openpni::experimental::node {
MichNormalization::MichNormalization(
    core::MichDefine __mich)
    : MichNormalization(std::make_unique<MichNormalization_impl>(__mich)) {}
MichNormalization::~MichNormalization() {}
MichNormalization::MichNormalization(
    std::unique_ptr<MichNormalization_impl> impl)
    : m_impl(std::move(impl)) {}
MichNormalization MichNormalization::copy() const {
  return MichNormalization(m_impl->copy());
}
std::unique_ptr<MichNormalization> MichNormalization::copyPtr() {
  return std::make_unique<MichNormalization>(m_impl->copy());
}

std::unique_ptr<float[]> MichNormalization::dumpNormalizationMich() {
  return m_impl->dumpNormalizationMich();
}

float const *MichNormalization::getHNormFactorsBatch(
    std::span<core::MichStandardEvent const> events, FactorBitMask im) {
  return m_impl->getHNormFactorsBatch(events, im);
}
float const *MichNormalization::getDNormFactorsBatch(
    std::span<core::MichStandardEvent const> events, FactorBitMask im) {
  return m_impl->getDNormFactorsBatch(events, im);
}
float const *MichNormalization::getHNormFactorsBatch(
    std::span<std::size_t const> lorIndices, FactorBitMask im) {
  return m_impl->getHNormFactorsBatch(lorIndices, im);
}
float const *MichNormalization::getDNormFactorsBatch(
    std::span<std::size_t const> lorIndices, FactorBitMask im) {
  return m_impl->getDNormFactorsBatch(lorIndices, im);
}
std::unique_ptr<float[]> MichNormalization::getActivityMich() {
  return m_impl->getActivityMich();
}
std::unique_ptr<float[]> MichNormalization::dumpCryFctMich() {
  return m_impl->dumpCryFctMich();
}
std::unique_ptr<float[]> MichNormalization::dumpBlockFctMich() {
  return m_impl->dumpBlockFctMich();
}
std::unique_ptr<float[]> MichNormalization::dumpRadialFctMich() {
  return m_impl->dumpRadialFctMich();
}
std::unique_ptr<float[]> MichNormalization::dumpPlaneFctMich() {
  return m_impl->dumpPlaneFctMich();
}
std::unique_ptr<float[]> MichNormalization::dumpInterferenceFctMich() {
  return m_impl->dumpInterferenceFctMich();
}
void MichNormalization::setShell(
    float innerRadius, float outerRadius, float axialLength, float parallaxScannerRadial, core::Grids<3, float> grids) {
  m_impl->setShell(innerRadius, outerRadius, axialLength, parallaxScannerRadial, grids);
}
void MichNormalization::setFActCorrCutLow(
    float v) {
  m_impl->setFActCorrCutLow(v);
}
void MichNormalization::setFActCorrCutHigh(
    float v) {
  m_impl->setFActCorrCutHigh(v);
}
void MichNormalization::setFCoffCutLow(
    float v) {
  m_impl->setFCoffCutLow(v);
}
void MichNormalization::setFCoffCutHigh(
    float v) {
  m_impl->setFCoffCutHigh(v);
}
void MichNormalization::setBadChannelThreshold(
    float v) {
  m_impl->setBadChannelThreshold(v);
}
void MichNormalization::setRadialModuleNumS(
    int v) {
  m_impl->setRadialModuleNumS(v);
}
void MichNormalization::bindComponentNormScanMich(
    float *promptMich) {
  m_impl->bindShellScanMich(promptMich);
}
void MichNormalization::bindComponentNormIdealMich(
    float *fwdMich) {
  m_impl->bindShellFwdMich(fwdMich);
}
void MichNormalization::bindSelfNormMich(
    float *delayMich) {
  m_impl->bindSelfNormMich(delayMich);
}
void MichNormalization::addSelfNormListmodes(
    std::span<basic::Listmode_t const> listmodes) {
  m_impl->addSelfNormListmodes(listmodes);
}
void MichNormalization::saveToFile(
    std::string path) {
  m_impl->saveToFile(path);
}
void MichNormalization::recoverFromFile(
    std::string path) {
  m_impl->recoverFromFile(path);
}
void MichNormalization::setDeadTimeTable(
    DeadTimeTable dtTable) {
  m_impl->setDeadTimeTable(dtTable);
}

} // namespace openpni::experimental::node
