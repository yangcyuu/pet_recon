#include <iostream>

#include "../public.hpp"
#include "include/experimental/example/SimplyRecon.hpp"
#include "include/experimental/node/MichScatter.hpp"
using namespace openpni::experimental;
const char *fileMichName = "shell_fwd.bin";
// const char *fileMichName = "/home/ustc/pni_test/0919test/testdata/coin_all.image3d";
const int fileMichOffset = 0;
// const int fileMichOffset = 6;
constexpr int minSectorDifference{4};
constexpr double taiFittingThreshold{0.95};
constexpr double scatterPointsThreshold{0.00124};
constexpr core::Vector<double, 3> scatterEnergyWindow{350.00, 650.00, 0.15};
constexpr core::Vector<double, 3> scatterEffTableEnergy{0.01, 700.00, 0.01};
int main() {
  auto e180 = E180();
  auto michInfo = core::MichInfoHub::create(e180);
  auto michData = read_from_file<float>(fileMichName, michInfo.getMichSize(), fileMichOffset);

  std::string in_attnMapFile =
      "/home/ustc/LGX_TEST/Data/20250826_recontestdata/FIN_WellCounter/CTRecon4888/ct_attn_img.dat";
  std::string in_attnCoffFile = "/home/ustc/LGX_TEST/test_0919/attnCoff.bin";
  std::string in_promptMichFile = "/home/ustc/LGX_TEST/Data/20250826_recontestdata/FIN_WellCounter/"
                                  "Slices_of_PET4643_2025-04-10-14-59-57/Slice4796/coin_all.image3d";
  std::string in_randomMichFile = "/home/ustc/LGX_TEST/test_0919/randCorrection.bin";
  std::string in_normMichFile = "/home/ustc/LGX_TEST/test_0919/sssNormCorrection.bin";
  std::string in_emapFile = "/home/ustc/LGX_TEST/test_0919/v3sss/osemForSSS.dat";

  auto michInfoHub = core::MichInfoHub::create(e180);
  auto attnMapGrids = core::Grids<3>::create_by_spacing_size(core::Vector<float, 3>::create(.5f, .5f, .5f),
                                                             core::Vector<int64_t, 3>::create(320, 320, 400));
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

  node::MichScatter scatter(e180);

  scatter.setMinSectorDifference(minSectorDifference);
  scatter.setTailFittingThreshold(taiFittingThreshold);
  scatter.setScatterPointsThreshold(scatterPointsThreshold);
  scatter.setScatterEnergyWindow(scatterEnergyWindow);
  scatter.setScatterEffTableEnergy(scatterEffTableEnergy);
  std::cout << "MichScatter parameters set" << std::endl;

  scatter.bindHAttnMap(attnMapGrids, h_attnMapData.get());
  scatter.bindHAttnCoff(h_attnCoffData.get());
  scatter.bindHNorm(h_normMichData.get());
  scatter.bindHRandom(h_randomMichData.get());
  scatter.bindHPromptMich(h_promptMichData.get());

  std::cout << "AttnMap,AttnCoff,Norm,EmissionMap,Random,PromptMich bound" << std::endl;

  auto michNorm = nullptr;
  auto michRand = nullptr;
  auto michScat = &scatter;
  std::cout << "read mich data done\n";

  example::OSEM_params params;
  params.binCutRatio = 0.20f;
  params.hfwhm = 1.0f;
  params.iterNum = 4;
  params.sample_rate = 0.5f;
  params.subsetNum = 12;
  params.scatterSimulations = 1;

  core::Grids<3, float> grids = core::Grids<3, float>::create_by_spacing_size(
      core::Vector<float, 3>::create(.5f, .5f, .5f), core::Vector<int64_t, 3>::create(320, 320, 400));
  std::unique_ptr<float[]> outImg = std::make_unique_for_overwrite<float[]>(grids.totalSize());
  example::instant_OSEM_mich_CUDA(core::Image3DOutput<float>{grids, outImg.get()}, params, michData.get(), michNorm,
                                  michRand, michScat, e180);
  std::cout << "OSEM done\n";
  std::ofstream outFile("shell_recon_img_gpu.bin", std::ios::binary);
  outFile.write(reinterpret_cast<char *>(outImg.get()), grids.totalSize() * sizeof(float));
  outFile.close();
}