#include <iostream>

#include "../public.hpp"
#include "include/experimental/example/SimplyRecon.hpp"

using namespace openpni::experimental;
int main() {
  std::filesystem::current_path("/media/ustc-pni/5282FE19AB6D5297/testCase/testE180Case_FBP");
  std::cout << "Current working directory: " << std::filesystem::current_path() << std::endl;

  std::cout << "\n=== FBP gpu Test ===\n" << std::endl;
  //==============================================================CONFIG LOADING
  std::string polygonDefineFilePath = "config/polygonSystemDefine.json";
  std::string fbpParamsFilePath = "config/fbpParams.json";
  std::string promptDataPath = "Data/source/coin_all.image3d";
  const int fileMichOffset = 6;

  openpni::experimental::core::MichDefine mich;
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
  auto michData = read_from_file<float>(promptDataPath, michInfo.getMichSize(), fileMichOffset);

  example::FBP_params params;
  auto fbpParams = readFromJson<openpni::autogen::json::FBPParams>(fbpParamsFilePath);
  {
    if (fbpParams.rebinMethod == "SSRB")
      params.rebinMethod = example::FBP_RebinMethod::SSRB;
    else if (fbpParams.rebinMethod == "FORE")
      params.rebinMethod = example::FBP_RebinMethod::FORE;
    else
      throw std::runtime_error("Unsupported rebin method: " + fbpParams.rebinMethod);
    params.nRingDiff = fbpParams.nRingDiff;
    params.nSampNumInBin = fbpParams.nSampNumInBin;
    params.nSampNumInView = fbpParams.nSampNumInView;
    params.deltalim = fbpParams.deltalim;
    params.klim = fbpParams.klim;
    params.wlim = fbpParams.wlim;
    params.sampling_distance_in_s = fbpParams.sampling_distance_in_s;
    params.detectorLen = fbpParams.detectorLen;
  }
  core::Grids<3, float> grids = core::Grids<3, float>::create_by_spacing_size(
      core::Vector<float, 3>::create(static_cast<float>(fbpParams.ImgGridsX), static_cast<float>(fbpParams.ImgGridsY),
                                     static_cast<float>(fbpParams.ImgGridsZ)),
      core::Vector<int64_t, 3>::create(fbpParams.ImgGridsNumX, fbpParams.ImgGridsNumY, fbpParams.ImgGridsNumZ));

  std::cout << "\n=== Configuration loaded successfully! ===\n" << std::endl;

  //===== recon
  std::unique_ptr<float[]> outImg = std::make_unique_for_overwrite<float[]>(grids.totalSize());
  example::instant_FBP_mich_CUDA(core::Image3DOutput<float>{grids, outImg.get()}, params, michData.get(), mich);
  std::cout << "FBP done\n";
  std::ofstream outFile("Data/result/fbp_recon_img_gpu.bin", std::ios::binary);
  outFile.write(reinterpret_cast<char *>(outImg.get()), grids.totalSize() * sizeof(float));
  outFile.close();

  return 0;
}