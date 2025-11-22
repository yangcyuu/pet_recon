#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "../public.hpp"
#include "include/experimental/example/SimplyRecon.hpp"
// const char *ct_data = "/media/ustc-pni/4E8CF2236FB7702F/Pni_code/fdk-cuda/pni-standard-project/test/data/"
//                       "case3753_ct_projections.img3d";
// const char *air_data = "/media/ustc-pni/4E8CF2236FB7702F/Pni_code/fdk-cuda/pni-standard-project/test/data/"
//                        "1_CBCT_1 Bed Normal_BED0.img3d";
using namespace openpni::experimental;
using namespace openpni;

int main() {
  std::cout << "\n=== FDK gpu Test ===\n" << std::endl;
  //==============================================================CONFIG LOADING
  std::string fdkParamsFilePath = "config/fdkParams.json";
  std::string geoCorrParamsFilePath = "config/GeoCorrParams.json";
  std::string ctDataPath = "Data/source/ct_projections.img3d";
  std::string airDataPath = "Data/source/1_CBCT_1 Bed Normal_BED0.img3d";
  const int fileMichOffset = 6;
  using CTRawDataType = io::U16Image;
  auto ctResult = CTRawDataType::loadFromFile(ctDataPath);

  // 先尝试加载 air 文件
  auto airResult = CTRawDataType::loadFromFile(airDataPath);

  auto ct = ctResult.value();
  auto geometry = ct.imageGeometry();
  auto airImg = CTRawDataType(geometry);
  auto airSpan = airImg.span();
  std::fill(airSpan.begin(), airSpan.end(), uint16_t(13500));
  airResult = airImg;

  example::FDK_params params;
  auto fdkParams = readFromJson<openpni::autogen::json::FDKParams>(fdkParamsFilePath);
  {
    // Check if geoCorrParamsFilePath exists, if not get geometry correction params from fdkParams
    std::ifstream geoCorrFile(geoCorrParamsFilePath);
    if (!geoCorrFile.good()) {
      params.geo_angle = fdkParams.Geo_angle;
      params.geo_offsetU = fdkParams.Geo_offsetU;
      params.geo_offsetV = fdkParams.Geo_offsetV;
      params.geo_SDD = fdkParams.Geo_SDD;
      params.geo_SOD = fdkParams.Geo_SOD;
    } else {
      std::cout << "Loading geometry correction parameters from: " << std::filesystem::absolute(geoCorrParamsFilePath)
                << std::endl;
      auto geoCorrParams = readFromJson<openpni::autogen::json::GeoCorrParams>(geoCorrParamsFilePath);
      params.geo_angle = geoCorrParams.Geo_angle;
      params.geo_offsetU = geoCorrParams.Geo_offsetU;
      params.geo_offsetV = geoCorrParams.Geo_offsetV;
      params.geo_SDD = geoCorrParams.Geo_SDD;
      params.geo_SOD = geoCorrParams.Geo_SOD;
    }
    params.fouriorCutoffLength = static_cast<unsigned>(fdkParams.FouriorCutoffLength);
    params.beamHardenParamA = fdkParams.BeamHardenParamA;
    params.beamHardenParamB = fdkParams.BeamHardenParamB;
    params.ct_slope = fdkParams.Ct_slope;
    params.ct_intercept = fdkParams.Ct_intercept;
    params.co_offset_x = fdkParams.Co_offset_x;
    params.co_offset_y = fdkParams.Co_offset_y;
    params.pixelSizeU = fdkParams.PixelSizeU;
    params.pixelSizeV = fdkParams.PixelSizeV;
  }
  core::Grids<3, float> grids = core::Grids<3, float>::create_by_spacing_size(
      core::Vector<float, 3>::create(static_cast<float>(fdkParams.ImgGridsX), static_cast<float>(fdkParams.ImgGridsY),
                                     static_cast<float>(fdkParams.ImgGridsZ)),
      core::Vector<int64_t, 3>::create(fdkParams.ImgGridsNumX, fdkParams.ImgGridsNumY, fdkParams.ImgGridsNumZ));

  std::cout << "\n=== Configuration loaded successfully! ===\n" << std::endl;

  //===== recon
  std::unique_ptr<float[]> outImg = std::make_unique_for_overwrite<float[]>(grids.totalSize());
  example::instant_FDK_CUDA(core::Image3DOutput<float>{grids, outImg.get()}, params, ctResult.value(),
                            airResult.value());
  std::cout << "FDK done\n";
  std::ofstream outFile("Data/result/ct_recon_img_gpu.bin", std::ios::binary);
  outFile.write(reinterpret_cast<char *>(outImg.get()), grids.totalSize() * sizeof(float));
  outFile.close();
}