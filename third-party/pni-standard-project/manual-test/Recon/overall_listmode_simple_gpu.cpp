#include <iostream>

#include "../public.hpp"
#include "include/experimental/example/SimplyRecon.hpp"
#include "include/experimental/node/GaussConv3D.hpp"
#include "src/experimental/impl/MichAttnImpl.hpp"
#include "src/experimental/impl/MichNormImpl.hpp"
#include "src/experimental/impl/Random.h"
using namespace openpni::experimental;
constexpr int minSectorDifference{4};
constexpr int radialModuleNumS{6};
constexpr float BadChannelThreshold{0.02};
constexpr double taiFittingThreshold{0.95};
constexpr double scatterPointsThreshold{0.00124};
constexpr core::Vector<double, 3> scatterEnergyWindow{350.00, 650.00, 0.15};
constexpr core::Vector<double, 3> scatterEffTableEnergy{0.01, 700.00, 0.01};

const auto listmodeFile = "/home/ustc/pni_test/old-coin-convert/test_listmode.pni";

int main() {
  tools::cpu_threads().setThreadNumType(tools::MAX_THREAD).setScheduleType(tools::DYNAMIC).setScheduleNum(64);

  auto polygonSys = E180();
  auto michInfo = core::MichInfoHub::create(polygonSys);
  // grids info
  auto attnGrids = core::Grids<3>::create_by_spacing_size(core::Vector<float, 3>::create(.5f, .5f, .5f),
                                                          core::Vector<int64_t, 3>::create(320, 320, 400));
  auto osemGrids = core::Grids<3>::create_by_spacing_size(core::Vector<float, 3>::create(.5f, .5f, .5f),
                                                          core::Vector<int64_t, 3>::create(320, 320, 400));
  // notice::no deadtime data so we dont do dt corrections

  // files
  std::string in_prompt = "/home/ustc/PETLAB-Software/LGX_TEST/Data/20250826_recontestdata/FIN_WellCounter/"
                          "Slices_of_PET4643_2025-04-10-14-59-57/Slice4796/coin_all.image3d";
  std::string in_delay = "/home/ustc/PETLAB-Software/LGX_TEST/Data/20250826_recontestdata/FIN_WellCounter/"
                         "Slices_of_PET4643_2025-04-10-14-59-57/Slice4796/delay_all.image3d";
  std::string in_attnMap =
      "/home/ustc/PETLAB-Software/LGX_TEST/Data/20250826_recontestdata/FIN_WellCounter/CTRecon4888/ct_attn_img.dat";

  auto promptData = read_from_file<float>(in_prompt, michInfo.getMichSize(), 6);
  auto delayData = read_from_file<float>(in_delay, michInfo.getMichSize(), 6);
  auto attnMapData = read_from_file<float>(in_attnMap, attnGrids.totalSize(), 6);

  //===== generateCorrections
  // 1.norm
  openpni::experimental::node::MichNormalization norm(polygonSys);
  // set norm parameters
  norm.recoverFromFile("/home/ustc/pni_core/new/pni-standard-project/manual-test/Normalization/test/norm_factors.dat");
  norm.bindSelfNormMich(delayData.get());
  // 2.attn
  openpni::experimental::node::MichAttn attn(polygonSys);
  attn.setFetchMode(node::MichAttn::FromPreBaked);
  // set attn parameters
  attn.setPreferredSource(node::MichAttn::Attn_GPU);
  attn.setMapSize(attnGrids);
  attn.bindHAttnMap(attnMapData.get());
  // 3.random
  openpni::experimental::node::MichRandom rand(polygonSys);
  // rand.setBadChannelThreshold(BadChannelThreshold); no badchannel in random
  rand.setMinSectorDifference(minSectorDifference);
  rand.setRadialModuleNumS(radialModuleNumS);
  rand.setDelayMich(delayData.get());
  // 4.scatter
  openpni::experimental::node::MichScatter scatter(polygonSys);
  scatter.setMinSectorDifference(minSectorDifference);
  scatter.setTailFittingThreshold(taiFittingThreshold);
  scatter.setScatterPointsThreshold(scatterPointsThreshold);
  scatter.setScatterEnergyWindow(scatterEnergyWindow);
  scatter.setScatterEffTableEnergy(scatterEffTableEnergy);
  scatter.bindAttnCoff(&attn);
  scatter.bindNorm(&norm);
  scatter.bindRandom(&rand);
  scatter.bindHPromptMich(promptData.get());
  scatter.bindHEmissionMap(osemGrids, nullptr);
  //===== recon
  example::OSEM_TOF_params params;
  params.binCutRatio = 0.f;
  params.iterNum = 4;
  params.sample_rate = 0.5f;
  params.subsetNum = 12;
  params.scatterSimulations = 1;
  params.size_GB = 4;

  node::GaussianConv3D conv3D;
  conv3D.setHWHM(1.0f); // 1.0 mm

  std::unique_ptr<float[]> outImg = std::make_unique_for_overwrite<float[]>(osemGrids.totalSize());
#define CALI 0
#if CALI
  example::instant_OSEM_listmode_CUDA(core::Image3DOutput<float>{osemGrids, outImg.get()}, params, conv3D, listmodeFile,
                                      4, &norm, &rand, nullptr, &attn, polygonSys);
#else
  example::instant_OSEM_listmode_CUDA(core::Image3DOutput<float>{osemGrids, outImg.get()}, params, conv3D, listmodeFile,
                                      nullptr, nullptr, nullptr, nullptr, polygonSys);
#endif

  std::cout << "OSEM done\n";

  std::ofstream outFile("overall_wellCounter_recon_img_gpu.bin", std::ios::binary);
  outFile.write(reinterpret_cast<char *>(outImg.get()), osemGrids.totalSize() * sizeof(float));
  outFile.close();

  return 0;
}