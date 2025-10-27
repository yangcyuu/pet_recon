#include <format>
#include <fstream>
#include <ranges>

#include "../public.hpp"
#include "include/experimental/node/MichRandom.hpp"
using namespace openpni::experimental;

void generateRandE180() {
  std::string in_delayfile = "";
  std::string in_delay = "/home/ustc/PETLAB-Software/LGX_TEST/Data/20250826_recontestdata/FIN_WellCounter/"
                         "Slices_of_PET4643_2025-04-10-14-59-57/Slice4796/delay_all.image3d";

  auto polygonSys = E180();
  auto michInfo = core::MichInfoHub::create(polygonSys);
  auto delayData = read_from_file<float>(in_delay, michInfo.getMichSize(), 6);
  // params
  int minSectorDifference = 4;
  int radialModuleNumS = 6;

  openpni::experimental::node::MichRandom rand(polygonSys);
  rand.setMinSectorDifference(minSectorDifference);
  rand.setRadialModuleNumS(radialModuleNumS);
  rand.setDelayMich(delayData.get());

  auto result = rand.dumpFactorsAsHMich();
  write_to_file("delay.dat", result.get(), michInfo.getMichSize());
}

int main() {
  generateRandE180();
  return 0;
}