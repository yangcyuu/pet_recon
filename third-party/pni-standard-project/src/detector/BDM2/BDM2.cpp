#include "include/detector/BDM2.hpp"

#include "BDM2_impl.hpp"

namespace openpni::device::bdm2 {
BDM2Runtime::BDM2Runtime() noexcept {
  m_impl = std::make_unique<bdm2::BDM2Runtime_impl>();
}
BDM2Runtime::~BDM2Runtime() noexcept {}

void BDM2Runtime::r2s_cpu() const noexcept {}

void BDM2Runtime::loadCalibration(
    const std::string filename) {
  m_impl->loadCaliTable(filename);
}

bool BDM2Runtime::isCalibrationLoaded() const noexcept {
  return m_impl->getisCalibrationLoaded();
}

DetectorUnchangable BDM2Runtime::detectorUnchangable() const noexcept {
  return openpni::device::detectorUnchangable<bdm2::BDM2Runtime>();
}

DetectorChangable &BDM2Runtime::detectorChangable() noexcept {
  return m_impl->detectorChangable();
}

const DetectorChangable &BDM2Runtime::detectorChangable() const noexcept {
  return m_impl->detectorChangable();
}

#if !PNI_STANDARD_CONFIG_DISABLE_CUDA
void BDM2Runtime::r2s_cuda(
    const void *d_raw, const device::PacketPositionInfo *d_position, uint64_t count,
    basic::LocalSingle_t *d_out) const noexcept {
  m_impl->r2s_cuda(d_raw, d_position, count, d_out);
}
#endif
} // namespace openpni::device::bdm2

namespace openpni::device::bdm2 {
BDM2Calibrater::BDM2Calibrater() noexcept {
  m_impl = std::make_unique<bdm2::BDM2Calibrater_impl>();
}

BDM2Calibrater::~BDM2Calibrater() noexcept {}

void BDM2Calibrater::appendData4Calibration(
    process::RawDataView rawdata, uint16_t channelIndex) {
  m_impl->appendData4Calibration(rawdata, channelIndex);
}

bool BDM2Calibrater::setCalibrationInitialValue(
    bdm2::KMeansInitialLine_t _x, bdm2::KMeansInitialLine_t _y, uint8_t duIndex) {
  return m_impl->setCalibrationInitialValue(_x, _y, duIndex);
}

bool BDM2Calibrater::setCalibrationInitialValue(
    bdm2::KMeansInitialValue_t _p, uint8_t duIndex) {
  return m_impl->setCalibrationInitialValue(_p, duIndex);
}

bool BDM2Calibrater::setCalibrationInitialValue(
    uint8_t duIndex) {
  return m_impl->setCalibrationInitialValue(duIndex);
}

bool BDM2Calibrater::generateCaliTable() {
  return m_impl->generateCaliTable();
}

const bdm2::CalibrationResult &BDM2Calibrater::getCalibrationResult() const {
  return m_impl->getCalibrationResult();
}

bdm2::CountMap_t BDM2Calibrater::countMap(
    uint8_t duIndex) {
  return m_impl->countMap(duIndex);
}

void BDM2Calibrater::loadCalibration(
    const std::string filename) {
  m_impl->loadCaliTable(filename);
}

void BDM2Calibrater::saveCalibration(
    const std::string filename) {
  m_impl->saveCaliTable(filename);
}

void BDM2Calibrater::adjustEnergyCoef(
    uint8_t duIndex, uint8_t crystalIndex, float peakEnergy) {
  m_impl->adjustEnergyCoef(duIndex, crystalIndex, peakEnergy);
}
} // namespace openpni::device::bdm2
