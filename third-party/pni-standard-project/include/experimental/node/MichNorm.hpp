#pragma once
#include <memory>

#include "../../IO.hpp"
#include "../core/Image.hpp"
#include "../core/Mich.hpp"
namespace openpni::experimental::node {
enum FactorBitMask : unsigned {
  CryFct = 1 << 0,
  BlockFct = 1 << 1,
  RadialFct = 1 << 2,
  PlaneFct = 1 << 3,
  InterferenceFct = 1 << 4,
  DTComponent = 1 << 5,
  All = 0xFFFFFFFF
};

struct DeadTimeTable;
class MichNormalization_impl;
class MichNormalization {
public:
  MichNormalization(core::MichDefine mich);
  MichNormalization(std::unique_ptr<MichNormalization_impl> impl);
  MichNormalization(MichNormalization &&) = default;
  ~MichNormalization();
  MichNormalization copy() const;
  std::unique_ptr<MichNormalization> copyPtr();

public:                                 // Preparing parameters
  void setShell(                        // 设置壳体的尺寸
      float innerRadius,                // 壳体内半径，单位mm
      float outerRadius,                // 壳体外半径，单位mm
      float axialLength,                // 壳体轴向长度，单位mm
      float parallaxScannerRadial,      // 视差效应下的扫描仪半径
      core::Grids<3, float> grids);     // 视差效应下的图像的网格
  void setFActCorrCutLow(float v);      // 0.05f
  void setFActCorrCutHigh(float v);     // 0.22f
  void setFCoffCutLow(float v);         // 0.0f
  void setFCoffCutHigh(float v);        // 100.0f
  void setBadChannelThreshold(float v); // 0.02f
  void setRadialModuleNumS(int v);      // 4
  void setDeadTimeTable(DeadTimeTable dtTable);
  void saveToFile(std::string path);
  void recoverFromFile(std::string path);
  void bindComponentNormScanMich(float *promptMich);
  void bindComponentNormIdealMich(float *fwdMich);
  void bindSelfNormMich(float *delayMich);
  void addSelfNormListmodes(std::span<basic::Listmode_t const> listmodes);

public:
  float const *getHNormFactorsBatch(std::span<core::MichStandardEvent const> events,
                                    FactorBitMask im = FactorBitMask::All);
  float const *getHNormFactorsBatch(std::span<std::size_t const> lorIndices, FactorBitMask im = FactorBitMask::All);
  float const *getDNormFactorsBatch(std::span<core::MichStandardEvent const> events,
                                    FactorBitMask im = FactorBitMask::All);
  float const *getDNormFactorsBatch(std::span<std::size_t const> lorIndices, FactorBitMask im = FactorBitMask::All);
  std::unique_ptr<float[]> dumpNormalizationMich();
  std::unique_ptr<float[]> getActivityMich();
  std::unique_ptr<float[]> dumpCryFctMich();
  std::unique_ptr<float[]> dumpBlockFctMich();
  std::unique_ptr<float[]> dumpRadialFctMich();
  std::unique_ptr<float[]> dumpPlaneFctMich();
  std::unique_ptr<float[]> dumpInterferenceFctMich();

private:
  std::unique_ptr<MichNormalization_impl> m_impl;
};

struct DeadTimeTable {
  std::vector<float> CTDTTable;
  std::vector<float> RTTable;
  double scanTime;
  double randomRateMin;
  float *delayMich;
};
} // namespace openpni::experimental::node
