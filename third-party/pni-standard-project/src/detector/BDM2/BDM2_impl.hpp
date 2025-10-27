#pragma once
#include <array>
#include <memory>

#include "include/basic/CudaPtr.hpp"
#include "include/basic/Point.hpp"
#include "include/detector/BDM2.hpp"
#include "include/detector/Detectors.hpp"
#include "src/autogen/autogen_xml.hpp"

namespace openpni::device::bdm2 {
class BDM2Runtime_impl {
public:
  BDM2Runtime_impl();
  ~BDM2Runtime_impl() = default;

public:
  void loadCaliTable(const std::string &filename);

public:
  DetectorChangable &detectorChangable() noexcept;
  const DetectorChangable &detectorChangable() const noexcept;

public:
  bool getisCalibrationLoaded() const noexcept { return isCalibrationLoaded; }

public:
#if !PNI_STANDARD_CONFIG_DISABLE_CUDA
  void r2s_cuda(const void *d_raw, const PacketPositionInfo *d_position, uint64_t count,
                basic::LocalSingle_t *d_out) const;
#endif
private:
  std::unique_ptr<CalibrationTable> m_h_caliTable;

#if !PNI_STANDARD_CONFIG_DISABLE_CUDA
  cuda_sync_ptr<float> m_d_energyCoef;
  cuda_sync_ptr<uint8_t> m_d_positionTable;
#endif

private:
  DetectorChangable m_detectorChangable;
  bool isCalibrationLoaded{false};
};

class BDM2Calibrater_impl {
public:
  BDM2Calibrater_impl();
  ~BDM2Calibrater_impl() = default;

public:
  void appendData4Calibration(process::RawDataView, uint16_t channelIndex);
  bool setCalibrationInitialValue(KMeansInitialLine_t _x, KMeansInitialLine_t _y, uint8_t duIndex);
  bool setCalibrationInitialValue(KMeansInitialValue_t, uint8_t duIndex);
  bool setCalibrationInitialValue(uint8_t duIndex);
  bool generateCaliTable();
  const bdm2::CalibrationResult &getCalibrationResult() const;
  CountMap_t countMap(uint8_t duIndex);

public:
  void saveCaliTable(const std::string &filename);
  void loadCaliTable(const std::string &filename);
  void adjustEnergyCoef(uint8_t duIndex, uint8_t crystalIndex, float newCoef);

public:
  bool getisCalibrationLoaded() const noexcept { return isCalibrationLoaded; }

public:
  DetectorChangable &detectorChangable() noexcept;
  const DetectorChangable &detectorChangable() const noexcept;

private:
  std::unique_ptr<CalibrationResult> m_calibrationResult;
  std::unique_ptr<CalibrationTable> m_h_caliTable;
  std::array<std::vector<TempFrameV2>, BLOCK_NUM> m_tempFrame;
  std::unique_ptr<std::array<CountMap_t, BLOCK_NUM>> m_countMap;
  std::unique_ptr<std::array<CenterPosition_t, BLOCK_NUM>> m_centerPosition;

private:
  DetectorChangable m_detectorChangable;
  bool isCalibrationLoaded{false};
};

} // namespace openpni::device::bdm2
