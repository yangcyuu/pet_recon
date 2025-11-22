#include <iostream>

#include "../public.hpp"
#include "include/experimental/example/SimplyRecon.hpp"
#include "include/experimental/node/GaussConv3D.hpp"
#include "include/experimental/node/KGConv3D.hpp"
#include "include/experimental/node/KNNConv3D.hpp"
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
int main() {
  tools::cpu_threads().setThreadNumType(tools::MAX_THREAD).setScheduleType(tools::DYNAMIC).setScheduleNum(64);
#define DEBUGPATH 0
#ifdef DEBUGPATH
  std::filesystem::current_path("/home/ustc/Desktop/testE180Case");
#endif
  std::cout << "Current working directory: " << std::filesystem::current_path() << std::endl;
  std::cout << "\n=== OSEM simple gpu Test ===\n" << std::endl;
  //==============================================================CONFIG LOADING
  std::string polygonDefineFilePath = "config/polygonSystemDefine.json";
  std::string osemParamsFilePath = "config/OSEMParams.json";
  std::string attnParamsFilePath = "config/attnParams.json";
  std::string normParamsFilePath = "config/normalizationParams.json";
  std::string scatParamsFilePath = "config/scatParams.json";
  std::string promptDataPath = "Data/source/coin_all.image3d";
  std::string delayDataPath = "Data/source/delay.pni";
  std::string normFactorPath = "Data/result/norm_factors.dat";
  std::string attnMapPath = "Data/source/attn_ct_img.dat";
  std::string kemImgInPath = "Data/result/image.pni";

  openpni::experimental::core::MichDefine mich;
  example::OSEM_params params;
  node::KGConv3D conv3D;
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

  auto kemImgGrid = core::Grids<3>::create_by_spacing_size(core::Vector<float, 3>::create(.5f, .5f, .5f),
                                                           core::Vector<int64_t, 3>::create(320, 320, 400));
  auto kemImgData = read_from_file<float>(kemImgInPath, kemImgGrid.totalSize(), 0);
  core::TensorDataInput<float, 3> kemImgTensorDataIn{kemImgGrid, kemImgData.get()};
  auto d_kemImgData = openpni::make_cuda_sync_ptr_from_hcopy(
      std::span<const float>(kemImgData.get(), kemImgGrid.totalSize()), "kemImgData");
  core::TensorDataInput<float, 3> d_kemImgTensorDataIn{kemImgGrid, d_kemImgData.get()};

  auto OSEMParams = readFromJson<openpni::autogen::json::michOSEMParams>(osemParamsFilePath);
  {
    params.subsetNum = static_cast<int>(OSEMParams.SubsetNum);
    params.iterNum = static_cast<int>(OSEMParams.IterNum);
    params.binCutRatio = static_cast<float>(OSEMParams.BinCutRatio);
    params.sample_rate = static_cast<float>(OSEMParams.SampleRate);
    params.scatterSimulations = static_cast<int>(OSEMParams.ScatterSimulations);
    conv3D.setHWHM(static_cast<float>(OSEMParams.ConvHWHM));
    conv3D.setKNNNumbers(30);
    conv3D.setFeatureSizeHalf(core::Vector<int64_t, 3>::create(1, 1, 1));
    conv3D.setKNNSearchSizeHalf(core::Vector<int64_t, 3>::create(2, 2, 2));
    conv3D.setKNNSigmaG2(32.f);
    conv3D.setDTensorDataIn(d_kemImgTensorDataIn);
  }

  auto osemGrids = core::Grids<3>::create_by_spacing_size(
      core::Vector<float, 3>::create(static_cast<float>(OSEMParams.ImgGridsX), static_cast<float>(OSEMParams.ImgGridsY),
                                     static_cast<float>(OSEMParams.ImgGridsZ)),
      core::Vector<int64_t, 3>::create(OSEMParams.ImgGridsNumX, OSEMParams.ImgGridsNumY, OSEMParams.ImgGridsNumZ));

  auto promptData = read_from_file<float>(promptDataPath, michInfo.getMichSize(), 6);
  std::unique_ptr<float[]> attnMapData;
  std::unique_ptr<float[]> delayData;

  if (OSEMParams.HasNorm) {
    auto normParams = readFromJson<openpni::autogen::json::NormalizationParams>(normParamsFilePath);
    norm = new node::MichNormalization(mich);
    norm->recoverFromFile(normFactorPath);
    delayData = read_from_file<float>(delayDataPath, michInfo.getMichSize(), 6);
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
    rand = new node::MichRandom(mich);
    rand->setMinSectorDifference(minSectorDifference);
    rand->setRadialModuleNumS(radialModuleNumS);
    delayData = read_from_file<float>(delayDataPath, michInfo.getMichSize(), 6);
    rand->setDelayMich(delayData.get());
  }
  if (OSEMParams.HasScat && attn != nullptr && norm != nullptr && rand != nullptr) {
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

  //===== recon
  std::unique_ptr<float[]> outImg = std::make_unique_for_overwrite<float[]>(osemGrids.totalSize());
#define CALI 1
#if CALI
  example::instant_OSEM_mich_CUDA(core::Image3DOutput<float>{osemGrids, outImg.get()}, params, conv3D, promptData.get(),
                                  norm, rand, scatter, attn, mich);
#else
  example::instant_OSEM_mich_CUDA(core::Image3DOutput<float>{osemGrids, outImg.get()}, params, conv3D, promptData.get(),
                                  nullptr, nullptr, nullptr, nullptr, mich);
#endif

  std::cout << "OSEM done\n";

  {
    auto d_img = openpni::make_cuda_sync_ptr_from_hcopy(std::span<float const>(outImg.get(), osemGrids.totalSize()),
                                                        "OSEM_outImg");
    conv3D.convD(openpni::experimental::core::TensorDataIO<float, 3>{osemGrids, d_img.get(), d_img.get()});
    d_img.allocator().copy_from_device_to_host(outImg.get(), d_img.cspan());
    for (const auto [x, y, z] : osemGrids.size) {
      auto radius = core::Vector<float, 2>::create(x, y) -
                    core::Vector<float, 2>::create(osemGrids.size.dimSize[0], osemGrids.size.dimSize[1]) * 0.5f;
      float r_len = algorithms::l2(radius);
      if (r_len > 100) {
        outImg[osemGrids.size(x, y, z)] = 0;
      }
    }
  }

  std::ofstream outFile("Data/result/overall_wellCounter_kem_recon_img_gpu.bin", std::ios::binary);
  outFile.write(reinterpret_cast<char *>(outImg.get()), osemGrids.totalSize() * sizeof(float));
  outFile.close();

  return 0;
}