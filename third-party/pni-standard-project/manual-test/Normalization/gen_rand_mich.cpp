#include <format>
#include <fstream>

#include "../public.hpp"
#include "include/experimental/node/MichRandom.hpp"
constexpr int minSectorDifference{4};
constexpr int radialModuleNumS{6};
constexpr float BadChannelThreshold{0.02};
using namespace openpni::experimental;
int main() {
  std::string in_delayFile = "/home/ustc/test_data/coin_all.image3d";
  std::string out_randFile = "randCorrection.bin";
  auto e180 = E180();

  auto michSize = core::MichInfoHub::create(e180).getMichSize();
  auto delayMich = read_from_file<float>(in_delayFile, michSize, 6);

  openpni::experimental::node::MichRandom norm(e180);
  norm.setBadChannelThreshold(BadChannelThreshold);
  norm.setRadialModuleNumS(radialModuleNumS);
  norm.setMinSectorDifference(minSectorDifference);
  norm.setDelayMich(delayMich.get());

  std::cout << "Doing random calculation..." << std::endl;
  auto randomMich = norm.dumpFactorsAsHMich();
  std::cout << "Random calculation done." << std::endl;
  write_to_file(out_randFile, randomMich.get(), michSize);
}