#pragma once
#include "../core/Image.hpp"
#include "MichAttn.hpp"
#include "MichNorm.hpp"
#include "MichRandom.hpp"
namespace openpni::experimental::node {

class MichScatter_impl;
class MichScatter {
public:
  MichScatter(core::MichDefine mich);
  MichScatter(std::unique_ptr<MichScatter_impl> impl);
  MichScatter(MichScatter &&) = default;
  ~MichScatter();
  MichScatter copy();
  std::unique_ptr<MichScatter> copyPtr();

public:
  void setScatterPointsThreshold(double v);
  void setTailFittingThreshold(double v);
  void setScatterEnergyWindow(core::Vector<double, 3> windows);
  void setScatterEnergyWindow(double low, double high, double resolution);
  void setScatterEffTableEnergy(core::Vector<double, 3> energies);
  void setScatterEffTableEnergy(double low, double high, double interval);
  void setMinSectorDifference(int v);
  void setScatterPointGrid(core::Grids<3> grid);
  void setTOFParams(double timeBinWidth, double timeBinStart, double timeBinEnd, double systemTimeRes_ns);

public:
  //====== inputs
  // The input pointers below only hold the access to the data, no copy inside.
  // If emission h_data or d_data is nullptr, it means a blank emission map (all zeros), then function
  // will ignore the map and set all scatter factors to zero.
  void bindAttnCoff(MichAttn *attnCoff);
  void bindNorm(MichNormalization *norm);
  void bindRandom(MichRandom *random);
  void bindHPromptMich(float *h_promptMich);
  void bindDPromptMich(float *d_promptMich);
  // void bindHListmode(std::span<basic::Listmode_t const> listmodes);
  void bindDListmode(std::span<basic::Listmode_t const> listmodes);
  void bindHEmissionMap(core::Grids<3, float> emap, float const *h_data);
  void bindDEmissionMap(core::Grids<3, float> emap, float const *d_data);

public:
  float const *getHScatterFactorsBatch(std::span<core::MichStandardEvent const> events);
  float const *getDScatterFactorsBatch(std::span<core::MichStandardEvent const> events);
  std::unique_ptr<float[]> dumpScatterMich();

private:
  std::unique_ptr<MichScatter_impl> m_impl;
};

} // namespace openpni::experimental::node
