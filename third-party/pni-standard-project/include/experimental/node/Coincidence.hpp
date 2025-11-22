#pragma once
#include "../../basic/PetDataType.h"
#include "../interface/SingleGenerator.hpp"
namespace openpni::experimental::node {

struct CoincidenceProtocol {
  int timeWindow_ps{2'000};
  int delayTime_ps{2'000'000};
  float energyLower_eV{350'000};
  float energyUpper_eV{650'000};
};
#pragma pack(push, 1)
struct LocalListmode {
  unsigned short channelIndex1;
  unsigned short crystalIndex1;
  unsigned short channelIndex2;
  unsigned short crystalIndex2;
  unsigned short time1_2pico;
};
#pragma pack(pop)

class Coincidence_impl;
class Coincidence {
public:
  Coincidence();
  ~Coincidence();

public:
  void setTotalCrystalNumOfEachChannel(std::vector<uint32_t> const &crystalNumPerChannel) const;

public:
  struct CoinResult {
    std::span<LocalListmode const> prompt;
    std::span<LocalListmode const> delay;
  };

  CoinResult getDListmode(std::vector<std::span<interface::LocalSingle const>> d_coinListmodeLists,
                          CoincidenceProtocol protocol) const;
  auto getDListmode(
      std::span<interface::LocalSingle const> d_coinListmode, CoincidenceProtocol protocol) {
    return this->getDListmode(std::vector{d_coinListmode}, protocol);
  };
  std::vector<unsigned> dumpCrystalCountMap() const;
  void clearCrystalCountMap() const;

private:
  std::unique_ptr<Coincidence_impl> m_impl;
};
} // namespace openpni::experimental::node
