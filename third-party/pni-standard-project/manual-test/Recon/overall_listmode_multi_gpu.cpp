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

const auto listmodeFile = "/home/deep02/SoftGroup/PNI_workGroup/testData/test_listmode.pni";

int main() {
  tools::cpu_threads().setThreadNumType(tools::MAX_THREAD).setScheduleType(tools::DYNAMIC).setScheduleNum(64);

  auto polygonSys = E180();
  auto michInfo = core::MichInfoHub::create(polygonSys);
  // grids info
  auto attnGrids = core::Grids<3>::create_by_spacing_size(core::Vector<float, 3>::create(.5f, .5f, .5f),
                                                          core::Vector<int64_t, 3>::create(320, 320, 400));
  auto osemGrids = core::Grids<3>::create_by_center_boxLength_size(core::Vector<float, 3>::create(0.f, 0.f, 0.f),
                                                                   core::Vector<float, 3>::create(160, 160, 200),
                                                                   core::Vector<int64_t, 3>::create(250, 250, 144));
  // notice::no deadtime data so we dont do dt corrections

  // files
  std::string in_prompt = "/home/deep02/SoftGroup/PNI_workGroup/testData/coin_all.image3d";
  std::string in_delay = "/home/deep02/SoftGroup/PNI_workGroup/testData/delay_all.image3d";
  std::string in_attnMap = "/home/deep02/SoftGroup/PNI_workGroup/testData/ct_attn_img.dat";
  std::string in_normFactors = "/home/deep02/SoftGroup/PNI_workGroup/testData/norm_factors.dat";

  auto promptData = read_from_file<float>(in_prompt, michInfo.getMichSize(), 6);
  auto delayData = read_from_file<float>(in_delay, michInfo.getMichSize(), 6);
  auto attnMapData = read_from_file<float>(in_attnMap, attnGrids.totalSize(), 6);

  example::OSEM_TOF_MULTI_params params;
  params.binCutRatio = 0.f;
  params.iterNum = 4;
  params.sample_rate = 0.5f;
  params.subsetNum = 12;
  params.scatterSimulations = 3;
  params.size_GB = 4;
  params.TOF_division = 500;
  params.randomListmodeFile = {"/home/deep02/SoftGroup/PNI_workGroup/testData/small_mouse_test_listmode.pni"};
  params.timeBegin_ms = 0;
  params.timeEnd_ms = 0xFFFFFFFF;
  params.minSectorDiff = minSectorDifference;
  params.randRadialModuleNumS = radialModuleNumS;
  params.doNorm = true;
  params.normFactorsFile = "/home/deep02/SoftGroup/PNI_workGroup/testData/norm_factors.dat";
  params.doSelfNorm = true;
  params.selfNormMich = delayData.get();
  params.doAttn = true;
  params.attnMap = core::Image3DInput<float>{attnGrids, attnMapData.get()}; // ct_attn_img.dat is in 0.5mm*0.5mm*0.5mm
  params.doScatter = true;
  params.tailFittingThreshold = taiFittingThreshold;
  params.scatterPointsThreshold = scatterPointsThreshold;
  params.scatterEnergyWindow = scatterEnergyWindow;
  params.scatterEffTableEnergy = scatterEffTableEnergy;
  params.gauss_hwhm_mm = 1.0f;
  params.listmode_paths = {listmodeFile};
  params.bigVoxelScatterSimulation = true;

  node::GaussianConv3D conv3D;
  conv3D.setHWHM(1.0f); // 1.0 mm

  std::unique_ptr<float[]> outImg = std::make_unique_for_overwrite<float[]>(osemGrids.totalSize());
#define CALI 1
#if CALI
  example::instant_OSEM_listmodeTOF_MULTI_CUDA(core::Image3DOutput<float>{osemGrids, outImg.get()}, params, polygonSys);
#else
  example::instant_OSEM_listmode_CUDA(core::Image3DOutput<float>{osemGrids, outImg.get()}, params, conv3D, listmodeFile,
                                      nullptr, nullptr, nullptr, nullptr, polygonSys);
#endif

  std::cout << "OSEM done\n";

  std::ofstream outFile("overall_wellCounter_recon_listmode_gpu.bin", std::ios::binary);
  outFile.write(reinterpret_cast<char *>(outImg.get()), osemGrids.totalSize() * sizeof(float));
  outFile.close();

  return 0;
}