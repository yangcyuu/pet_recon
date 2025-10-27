#pragma once
#include <array>
#include <memory>
#include <vector>

#include "../basic/CudaPtr.hpp"
#include "../basic/Point.hpp"
#include "../process/Acquisition.hpp"
#include "Detectors.hpp"

namespace openpni::device {
namespace bdm2 {
#pragma pack(push, 1)
typedef struct DataFrameV2 {
  uint8_t nHeadAndDU;
  uint8_t nBDM;
  uint8_t nTime[8];
  uint8_t X;
  uint8_t Y;
  uint8_t Energy[2];
  int8_t nTemperatureInt;
  uint8_t nTemperatureAndTail;
} DataFrameV2;
#pragma pack(pop)

constexpr uint64_t UDP_PACKET_SIZE = 1024;
constexpr uint64_t MAX_UDP_PACKET_SIZE = UDP_PACKET_SIZE;
constexpr uint64_t MIN_UDP_PACKET_SIZE = UDP_PACKET_SIZE;

constexpr uint64_t SINGLE_NUM_PER_PACKET = MAX_UDP_PACKET_SIZE / sizeof(DataFrameV2);
constexpr uint64_t MAX_SINGLE_NUM_PER_PACKET = SINGLE_NUM_PER_PACKET;
constexpr uint64_t MIN_SINGLE_NUM_PER_PACKET = SINGLE_NUM_PER_PACKET;

constexpr uint64_t CRYSTAL_LINE = 13;
constexpr uint64_t CRYSTAL_RAW_POSITION_RANGE = 256;
constexpr uint64_t CRYSTAL_NUM_ONE_BLOCK = CRYSTAL_LINE * CRYSTAL_LINE;
constexpr uint64_t BLOCK_NUM = 4;
constexpr uint64_t CRYSTAL_NUM = BLOCK_NUM * CRYSTAL_NUM_ONE_BLOCK;
constexpr float BLOCK_PITCH = 26.5;
constexpr float CRYSTAL_SIZE = 1.89;
constexpr float CRYSTAL_PITCH = 2.0;

struct TempFrameV2 {
  unsigned char x;
  unsigned char y;
  unsigned char du;
  float energy;
};
}; // namespace bdm2

namespace bdm2 {
class BDM2Runtime_impl;
class BDM2Calibrater_impl;

}; // namespace bdm2

namespace bdm2 {
using EnergyCoefs_t = std::array<float, CRYSTAL_NUM_ONE_BLOCK>;
using PositionTable_t = std::array<uint8_t, 256 * 256>;
using CenterPosition_t = std::array<basic::Vec2<float>, CRYSTAL_NUM_ONE_BLOCK>;
using CountMap_t = std::array<unsigned, 256 * 256>;
using KMeansInitialValue_t = std::array<basic::Vec2<float>, CRYSTAL_NUM_ONE_BLOCK>;
using KMeansInitialLine_t = std::array<uint8_t, CRYSTAL_LINE>;
using EnergyProfile_t = std::array<float, 1024>;
using EnergyProfileCut_t = std::array<float, 2>;

struct CalibrationResult {
  std::unique_ptr<std::array<EnergyCoefs_t, BLOCK_NUM>>
      energyCoef; // 每个晶体的能量系数，原始能量值乘上这个系数在511keV附近
  std::unique_ptr<std::array<PositionTable_t, BLOCK_NUM>>
      positionTable; // 每个原始位置:[0,255] ** [0,255] 映射到真实晶体索引: [0,13*13)
  std::unique_ptr<std::array<CountMap_t, BLOCK_NUM>> countMap; // 每个原始位置对应的计数统计，统计每个位置的事件数量
  std::unique_ptr<std::array<CenterPosition_t, BLOCK_NUM>> centerPosition; // 每个原始位置对应的晶体所在的原始位置
  std::unique_ptr<std::array<std::array<EnergyProfile_t, CRYSTAL_NUM_ONE_BLOCK>, BLOCK_NUM>>
      energyProfile; // 每个晶体的能谱统计
  std::unique_ptr<std::array<std::array<EnergyProfileCut_t, CRYSTAL_NUM_ONE_BLOCK>, BLOCK_NUM>>
      energyProfileCut; // 每个晶体的能谱统计切割值
};

struct CalibrationTable {
  std::unique_ptr<std::array<EnergyCoefs_t, BLOCK_NUM>> energyCoef;
  std::unique_ptr<std::array<PositionTable_t, BLOCK_NUM>> positionTable;
};

class BDM2Runtime final : public DetectorBase {
public:
  BDM2Runtime() noexcept;
  virtual ~BDM2Runtime() noexcept;

public:
  virtual void r2s_cpu() const noexcept override;

public:
  virtual void loadCalibration(const std::string filename) override;
  virtual DetectorUnchangable detectorUnchangable() const noexcept override;
  virtual DetectorChangable &detectorChangable() noexcept override;
  virtual const DetectorChangable &detectorChangable() const noexcept override;
  virtual bool isCalibrationLoaded() const noexcept override;
  virtual const char *detectorType() const noexcept override { return names::BDM2; }

#if !PNI_STANDARD_CONFIG_DISABLE_CUDA
  virtual void r2s_cuda(const void *d_raw, const PacketPositionInfo *d_position, uint64_t count,
                        basic::LocalSingle_t *d_out) const noexcept override;
#endif
private:
  std::unique_ptr<bdm2::BDM2Runtime_impl> m_impl;
};
class BDM2Calibrater {
public:
  BDM2Calibrater() noexcept;
  virtual ~BDM2Calibrater() noexcept;

public:
  void loadCalibration(const std::string filename);
  void saveCalibration(const std::string filename);

public:
  void appendData4Calibration(process::RawDataView rawdata, uint16_t channelIndex);
  // bool isCalibrationLoaded() const noexcept override { return false; }
  bool setCalibrationInitialValue(bdm2::KMeansInitialLine_t _x, bdm2::KMeansInitialLine_t _y, uint8_t duIndex);

  bool setCalibrationInitialValue(bdm2::KMeansInitialValue_t _p, uint8_t duIndex);

  bool setCalibrationInitialValue(uint8_t duIndex);

  bool generateCaliTable();

  const bdm2::CalibrationResult &getCalibrationResult() const;

  bdm2::CountMap_t countMap(uint8_t duIndex);

  void adjustEnergyCoef(uint8_t duIndex, uint8_t crystalIndex, float peakEnergy);

private:
  std::unique_ptr<bdm2::BDM2Calibrater_impl> m_impl;
};
}; // namespace bdm2
} // namespace openpni::device
// namespace openpni::device
