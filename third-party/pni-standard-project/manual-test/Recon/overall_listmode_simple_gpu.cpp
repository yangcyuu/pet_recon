#include <iostream>

#include "../public.hpp"
#include "include/experimental/example/SimplyRecon.hpp"
#include "include/experimental/node/GaussConv3D.hpp"
#include "src/experimental/impl/MichAttnImpl.hpp"
#include "src/experimental/impl/MichNormImpl.hpp"
#include "src/experimental/impl/Random.h"
using namespace openpni::experimental;

int main() {
  tools::cpu_threads().setThreadNumType(tools::MAX_THREAD).setScheduleType(tools::DYNAMIC).setScheduleNum(64);

#define DEBUGPATH 1
#ifdef DEBUGPATH
  std::filesystem::current_path("/media/ustc-pni/5282FE19AB6D5297/testCase/test930TOF");
#endif
  std::cout << "Current working directory: " << std::filesystem::current_path() << std::endl;

  std::cout << "\n=== OSEM  LM TOF simple gpu Test ===\n" << std::endl;
  //==============================================================CONFIG LOADING
  std::string polygonDefineFilePath = "config/polygonSystemDefine.json";
  std::string osemLMTOFParamsFilePath = "config/OSEM_LMTOFParams.json";
  std::string attnParamsFilePath = "config/attnParams.json";
  std::string normParamsFilePath = "config/normalizationParams.json";
  std::string scatParamsFilePath = "config/scatParams.json";
  std::string randParamsFilePath = "config/randomParams.json";
  std::string promptDataPath = "Data/source/coin_all.image3d";
  std::string listmodeFilePath = "Data/source/lm.pni";
  std::string delayDataPath = "Data/source/delay.pni";
  std::string normFactorPath = "Data/result/norm_factors.dat";
  std::string attnMapPath = "Data/source/attn_ct_img.dat";

  openpni::experimental::core::MichDefine mich;
  example::OSEM_TOF_params params;
  node::GaussianConv3D conv3D;
  openpni::experimental::node::MichNormalization *norm = nullptr;
  openpni::experimental::node::MichAttn *attn = nullptr;
  openpni::experimental::node::MichRandom *rand = nullptr;
  openpni::experimental::node::MichScatter *scatter = nullptr;
  {
    auto polyJson = readFromJson<openpni::autogen::json::PolygonalSystem>(polygonDefineFilePath);
    auto &polygon = mich.polygon;
    polygon.edges = polyJson.Edges;
    polygon.detectorLen = polyJson.DetectorLen;
    polygon.detectorPerEdge = polyJson.DetectorPerEdge;
    polygon.radius = polyJson.Radius;
    polygon.angleOf1stPerp = polyJson.AngleOf1stPerp;
    polygon.detectorRings = polyJson.DetectorRings;
    polygon.ringDistance = polyJson.RingDistance;
    auto &detector = mich.detector;
    detector.blockNumU = polyJson.DetectorBlockNumU;
    detector.blockNumV = polyJson.DetectorBlockNumV;
    detector.blockSizeU = polyJson.DetectorBlockSizeU;
    detector.blockSizeV = polyJson.DetectorBlockSizeV;
    detector.crystalNumU = polyJson.DetectorCrystalNumU;
    detector.crystalNumV = polyJson.DetectorCrystalNumV;
    detector.crystalSizeU = polyJson.DetectorCrystalSizeU;
    detector.crystalSizeV = polyJson.DetectorCrystalSizeV;
  }
  auto michInfo = core::MichInfoHub::create(mich);

  auto OSEMParams = readFromJson<openpni::autogen::json::michOSEM_LMTOFParams>(osemLMTOFParamsFilePath);
  {
    params.subsetNum = static_cast<int>(OSEMParams.SubsetNum);
    params.iterNum = static_cast<int>(OSEMParams.IterNum);
    params.binCutRatio = static_cast<float>(OSEMParams.BinCutRatio);
    params.sample_rate = static_cast<float>(OSEMParams.SampleRate);
    params.scatterSimulations = static_cast<int>(OSEMParams.ScatterSimulations);
    params.size_GB = static_cast<int>(OSEMParams.Size_GB);
    params.TOF_division = static_cast<int16_t>(OSEMParams.TOF_division);
    params.TOFBinWid_ps = static_cast<int16_t>(OSEMParams.TOFBinWid_ps);
    params.listmodeFileTimeBegin_ms = static_cast<int>(OSEMParams.ListmodeFileTimeBegin_ms);
    params.listmodeFileTimeEnd_ms = static_cast<int>(OSEMParams.ListmodeFileTimeEnd_ms);
    params.randomListmodeFile = OSEMParams.RandomListmodeFile;
    conv3D.setHWHM(static_cast<float>(OSEMParams.ConvHWHM));
  }

  auto osemGrids = core::Grids<3>::create_by_spacing_size(
      core::Vector<float, 3>::create(static_cast<float>(OSEMParams.ImgGridsX), static_cast<float>(OSEMParams.ImgGridsY),
                                     static_cast<float>(OSEMParams.ImgGridsZ)),
      core::Vector<int64_t, 3>::create(OSEMParams.ImgGridsNumX, OSEMParams.ImgGridsNumY, OSEMParams.ImgGridsNumZ));

  std::unique_ptr<float[]> promptData;
  std::unique_ptr<float[]> delayData;
  std::unique_ptr<float[]> attnMapData;

  if (OSEMParams.HasNorm || OSEMParams.HasRand) {
    delayData = read_from_file<float>(delayDataPath, michInfo.getMichSize(), 6);
  }
  if (OSEMParams.HasNorm) {
    auto normParams = readFromJson<openpni::autogen::json::NormalizationParams>(normParamsFilePath);
    norm = new node::MichNormalization(mich);
    norm->recoverFromFile(normFactorPath);
    norm->bindSelfNormMich(delayData.get());
    // if (OSEMParams.HasDeadtime) {
    //   norm->setDeadTimeTable(nullptr);
    // }
  }
  if (OSEMParams.HasAttn) {
    auto attnParams = readFromJson<openpni::autogen::json::AttnParams>(attnParamsFilePath);
    auto attnGrids = core::Grids<3>::create_by_spacing_size(
        core::Vector<float, 3>::create(attnParams.UmapGridsX, attnParams.UmapGridsY, attnParams.UmapGridsZ),
        core::Vector<int64_t, 3>::create(attnParams.UmapGridsNumX, attnParams.UmapGridsNumY, attnParams.UmapGridsNumZ));
    attnMapData = read_from_file<float>(attnMapPath, attnGrids.totalSize(), 6);
    attn = new node::MichAttn(mich);
    attn->setFetchMode(node::MichAttn::FromPreBaked);
    attn->setPreferredSource(node::MichAttn::Attn_GPU);
    attn->setMapSize(attnGrids);
    attn->bindHAttnMap(attnMapData.get());
  }
  if (OSEMParams.HasRand) {
    auto randParams = readFromJson<openpni::autogen::json::RandomParams>(randParamsFilePath);
    rand = new node::MichRandom(mich);
    rand->setMinSectorDifference(randParams.minSectorDifference);
    rand->setRadialModuleNumS(randParams.radialModuleNumS);
    rand->setDelayMich(delayData.get());
  }
  if (OSEMParams.HasScat && attn != nullptr && norm != nullptr && rand != nullptr) {
    promptData = read_from_file<float>(promptDataPath, michInfo.getMichSize(), 6);
    scatter = new node::MichScatter(mich);
    auto scatParams = readFromJson<openpni::autogen::json::scatParams>(scatParamsFilePath);
    auto sssGrids = core::Grids<3>::create_by_spacing_size(
        core::Vector<float, 3>::create(static_cast<float>(scatParams.SSSGridsX),
                                       static_cast<float>(scatParams.SSSGridsY),
                                       static_cast<float>(scatParams.SSSGridsZ)),
        core::Vector<int64_t, 3>::create(scatParams.SSSGridsNumX, scatParams.SSSGridsNumY, scatParams.SSSGridsNumZ));
    scatter->setMinSectorDifference(scatParams.MinSectorDifference);
    scatter->setTailFittingThreshold(scatParams.TaiFittingThreshold);
    scatter->setScatterPointsThreshold(scatParams.ScatterPointsThreshold);
    scatter->setScatterEnergyWindow(core::Vector<double, 3>::create(
        static_cast<double>(scatParams.ScatterEnergyWindowLow), static_cast<double>(scatParams.ScatterEnergyWindowHigh),
        static_cast<double>(scatParams.ScatterEnergyWindowResolution)));
    scatter->setScatterEffTableEnergy(
        core::Vector<double, 3>::create(static_cast<double>(scatParams.ScatterEffTableEnergyLow),
                                        static_cast<double>(scatParams.ScatterEffTableEnergyHigh),
                                        static_cast<double>(scatParams.ScatterEffTableEnergyInterval)));
    scatter->setScatterPointGrid(sssGrids);
    scatter->bindAttnCoff(attn);
    scatter->bindNorm(norm);
    scatter->bindRandom(rand);
    scatter->bindHPromptMich(promptData.get());
    scatter->bindHEmissionMap(osemGrids, nullptr);
  }

  std::cout << "\n=== Configuration loaded successfully! ===\n" << std::endl;

  std::unique_ptr<float[]> outImg = std::make_unique_for_overwrite<float[]>(osemGrids.totalSize());
#define CALI 1
#if CALI
  example::instant_OSEM_listmodeTOF_CUDA(core::Image3DOutput<float>{osemGrids, outImg.get()}, params, conv3D,
                                         listmodeFilePath, norm, rand, scatter, attn, mich);
#else
  example::instant_OSEM_listmode_CUDA(core::Image3DOutput<float>{osemGrids, outImg.get()}, params, conv3D,
                                      listmodeFilePath, nullptr, nullptr, nullptr, nullptr, mich);
  // example::instant_OSEM_listmodeTOF_CUDA(core::Image3DOutput<float>{osemGrids, outImg.get()}, params, conv3D,
  //                                     listmodeFilePath, nullptr, nullptr, nullptr, nullptr, mich);
#endif

  std::cout << "OSEM done\n";

  std::ofstream outFile("Data/result/overall_wellCounter_recon_img_gpu.bin", std::ios::binary);
  outFile.write(reinterpret_cast<char *>(outImg.get()), osemGrids.totalSize() * sizeof(float));
  outFile.close();

  return 0;
}