#include <iostream>

#include "../public.hpp"
#include "include/experimental/example/SimplyRecon.hpp"
#include "include/experimental/node/GaussConv3D.hpp"
#include "src/experimental/impl/MichAttnImpl.hpp"

using namespace openpni::experimental;

int main() {
  tools::cpu_threads().setThreadNumType(tools::MAX_THREAD).setScheduleType(tools::DYNAMIC).setScheduleNum(64);

  std::filesystem::current_path("/media/ustc-pni/5282FE19AB6D5297/testCase/test930TOF");
  std::cout << "Current working directory: " << std::filesystem::current_path() << std::endl;

  std::cout << "\n=== OSEM simple gpu Test ===\n" << std::endl;
  //==============================================================CONFIG LOADING
  std::string polygonDefineFilePath = "config/polygonSystemDefine.json";
  std::string osemLMTOFParamsFilePath = "config/OSEM_LMTOFParams.json";
  std::string listmodeFilePath = "Data/source/lm.pni";

  openpni::experimental::core::MichDefine mich;
  example::OSEM_TOF_params params;
  node::GaussianConv3D conv3D;
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

  std::cout << "\n=== Configuration loaded successfully! ===\n" << std::endl;

  std::unique_ptr<float[]> outImg = std::make_unique_for_overwrite<float[]>(osemGrids.totalSize());

  example::instant_backwardProjection_listmode_CUDA(core::Image3DOutput<float>{osemGrids, outImg.get()}, params, conv3D,
                                                    listmodeFilePath, mich);

  std::cout << "backwardProjection done\n";

  std::ofstream outFile("Data/result/LMbackwardProjection_img_gpu.bin", std::ios::binary);
  outFile.write(reinterpret_cast<char *>(outImg.get()), osemGrids.totalSize() * sizeof(float));
  outFile.close();

  return 0;
}