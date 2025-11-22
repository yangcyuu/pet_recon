#pragma once
#include "../../IO.hpp"
#include "../../basic/CudaPtr.hpp"
#include "../core/Mich.hpp"
namespace openpni::experimental::node {
class MichRandom_impl;
class MichRandom {
public:
  MichRandom(const core::MichDefine &mich);
  MichRandom(std::unique_ptr<MichRandom_impl> impl);
  MichRandom(MichRandom &&) noexcept;
  MichRandom &operator=(MichRandom &&) noexcept;
  ~MichRandom();
  MichRandom copy() const;
  std::unique_ptr<MichRandom> copyPtr();

public:
  // void setRandomRatio(float);
  //  lgxtest
  void setCountRatio(float countRatio);
  void setTimeBinRatio(float timeBinRatio);
  float getCountRatio();
  float getTimeBinRatio();
  //

  void setMinSectorDifference(int minSectorDifference);
  void setRadialModuleNumS(int radialModuleNumS);
  void setBadChannelThreshold(float badChannelThreshold);
  void setDelayMich(float *delayMich);
  void addDelayMich(float *delayMich);
  void addDelayListmodes(std::span<basic::Listmode_t const> listmodeFiles);
  std::vector<float> const &getFactors();
  std::unique_ptr<float[]> dumpFactorsAsHMich();
  cuda_sync_ptr<float> dumpFactorsAsDMich();
  void saveToFile(std::string path);
  void loadFromFile(std::string path);
  float const *getHRandomFactorsBatch(std::span<core::MichStandardEvent const> events);
  float const *getHRandomFactorsBatch(std::span<std::size_t const> lorIndices);
  float const *getDRandomFactorsBatch(std::span<core::MichStandardEvent const> events);
  float const *getDRandomFactorsBatch(std::span<std::size_t const> lorIndices);

private:
  std::unique_ptr<MichRandom_impl> m_impl;
};
} // namespace openpni::experimental::node
