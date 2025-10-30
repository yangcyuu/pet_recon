#include <format>
#include <fstream>

#include "../public.hpp"
#include "src/experimental/impl/MichNormImpl.hpp"
constexpr float fActCorrCutLow{0.05};
constexpr float fActCorrCutHigh{0.22};
constexpr float fCoffCutLow{0.0};
constexpr float fCoffCutHigh{100.0};
constexpr float BadChannelThreshold{0.02};
constexpr int radialModuleNumS{4};
int main() {
  std::string in_delayFile = "/home/ustc/LGX_TEST/Data/20250826_recontestdata/FIN_WellCounter/"
                             "Slices_of_PET4643_2025-04-10-14-59-57/Slice4796/delay_all.image3d";

  auto e180 = E180();

  auto normData = read_from_file<float>(in_delayFile, e180.michInfoHub().getMichSize(), 6);
  std::cout << std::format("read delay mich done, size = {}\n", e180.michInfoHub().getMichSize());

  openpni::experimental::node::MichNormalization_impl norm(e180);

  std::cout << "Doing normalization calculation..." << std::endl;
#define DUMP(filename, action)                                                                                         \
  {                                                                                                                    \
    write_to_file(filename, norm.action(), e180.michInfoHub().getMichSize());                                          \
    std::cout << std::format("Dump {} done.\n", filename);                                                             \
  }

  norm.recoverFromFile("norm_factors.dat");
  std::cout << "Recover from norm_factors.dat done." << std::endl;
  DUMP("norm_factors_recover.bin", dumpNormalizationMich);

  norm.bindSelfNormMich(normData.get());
  DUMP("self_norm.bin", dumpNormalizationMich);

#undef DUMP
}