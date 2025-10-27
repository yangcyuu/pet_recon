#include <format>
#include <fstream>

#include "../public.hpp"
#include "src/experimental/impl/MichAttnImpl.hpp"
#include "src/experimental/impl/MichNormImpl.hpp"
#include "src/experimental/impl/MichScatterImpl.hpp"
using namespace openpni::experimental;
constexpr int minSectorDifference{4};
constexpr double taiFittingThreshold{0.95};
constexpr double scatterPointsThreshold{0.00124};
constexpr core::Vector<double, 3> scatterEnergyWindow{350.00, 650.00, 0.15};
constexpr core::Vector<double, 3> scatterEffTableEnergy{0.01, 700.00, 0.01};
int main() {
  std::string in_attnMapFile =
      "/home/ustc/LGX_TEST/Data/20250826_recontestdata/FIN_WellCounter/CTRecon4888/ct_attn_img.dat";
  std::string in_attnCoffFile = "/home/ustc/LGX_TEST/test_0919/attnCoff.bin";
  std::string in_promptMichFile = "/home/ustc/LGX_TEST/Data/20250826_recontestdata/FIN_WellCounter/"
                                  "Slices_of_PET4643_2025-04-10-14-59-57/Slice4796/coin_all.image3d";
  std::string in_randomMichFile = "/home/ustc/LGX_TEST/test_0919/randCorrection.bin";
  std::string in_normMichFile = "/home/ustc/LGX_TEST/test_0919/sssNormCorrection.bin";
  std::string in_emapFile = "/home/ustc/LGX_TEST/test_0919/v3sss/osemForSSS.dat";

  auto polygonSys = E180();
  auto michInfoHub = core::MichInfoHub::create(polygonSys);
  auto attnMapGrids = core::Grids<3>::create_by_spacing_size(core::Vector<float, 3>::create(.5f, .5f, .5f),
                                                             core::Vector<int64_t, 3>::create(320, 320, 400));
  auto emapGrids = core::Grids<3>::create_by_spacing_size(core::Vector<float, 3>::create(2.f, 2.f, 2.f),
                                                          core::Vector<int64_t, 3>::create(80, 80, 100));
  auto h_attnMapData = read_from_file<float>(in_attnMapFile, attnMapGrids.totalSize(), 6);
  std::cout << std::format("read attnMap data done, size = {}\n", attnMapGrids.totalSize());
  auto h_attnCoffData = read_from_file<float>(in_attnCoffFile, michInfoHub.getMichSize(), 0);
  std::cout << std::format("read attnCoff data done, size = {}\n", michInfoHub.getMichSize());
  auto h_promptMichData = read_from_file<float>(in_promptMichFile, michInfoHub.getMichSize(), 6);
  std::cout << std::format("read promptMich data done, size = {}\n", michInfoHub.getMichSize());
  auto h_randomMichData = read_from_file<float>(in_randomMichFile, michInfoHub.getMichSize(), 0);
  std::cout << std::format("read randomMich data done, size = {}\n", michInfoHub.getMichSize());
  auto h_normMichData = read_from_file<float>(in_normMichFile, michInfoHub.getMichSize(), 0);
  std::cout << std::format("read normMich data done, size = {}\n", michInfoHub.getMichSize());
  auto h_emapData = read_from_file<float>(in_emapFile, emapGrids.totalSize(), 0);
  std::cout << std::format("read emap data done, size = {}\n", emapGrids.totalSize());

  node::MichScatter_impl scatter(polygonSys);
  node::MichAttn_impl attn(polygonSys);
  node::MichNormalization_impl norm(polygonSys);

  attn.test_bindHExistingAttn(h_attnCoffData.get(), h_attnMapData.get(), attnMapGrids);

  scatter.setMinSectorDifference(minSectorDifference);
  scatter.setTailFittingThreshold(taiFittingThreshold);
  scatter.setScatterPointsThreshold(scatterPointsThreshold);
  scatter.setScatterEnergyWindow(scatterEnergyWindow);
  scatter.setScatterEffTableEnergy(scatterEffTableEnergy);
  std::cout << "MichScatter parameters set" << std::endl;

  scatter.bindAttnCoff(attnMapGrids, h_attnMapData.get());
  scatter.bindNorm(h_normMichData.get());
  scatter.bindRandom(h_randomMichData.get());
  scatter.bindHEmissionMap(emapGrids, h_emapData.get());
  scatter.bindHPromptMich(h_promptMichData.get());

  std::cout << "AttnMap,AttnCoff,Norm,EmissionMap,Random,PromptMich bound" << std::endl;
  std::cout << "do sss" << std::endl;
  auto result = scatter.dumpHScatterMich(); // This will generate the scatter mich data
  std::cout << "Done,saving" << std::endl;
  write_to_file(std::format("scat_mich_{}.dat", "scatMich.bin"), result, michInfoHub.getMichSize());
}
