#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "../../include/io/IO.hpp"
#include "../../include/process/GeoCorr.hpp"
#include "../public.hpp"

using json = nlohmann::json;
using namespace openpni;
using namespace openpni::process;

int main() {
  const int proj_size_u = 1536;
  const int proj_size_v = 972;
  const int proj_num = 400;
  const float pixel_size_u = 0.1496f;
  const float pixel_size_v = 0.1496f;
  const int ball_num = 8;
  const float ball_distance = 3.0f;
  const float threshold_percent = 0.25f;
  std::string ctDataPath = "Data/source/CBCT_Geo_BED0.img3d";

  using CTRawDataType = io::U16Image;
  auto ctResult = CTRawDataType::loadFromFile(ctDataPath);
  auto &ctImage = ctResult.value();
  auto projPixelSize = ctImage.imageGeometry().voxelNum.x * ctImage.imageGeometry().voxelNum.y;
  std::vector<const CTRawDataType::value_type *> projectionPtrs;
  projectionPtrs.reserve(ctImage.imageGeometry().voxelNum.z);
  for (size_t i = 0; i < ctImage.imageGeometry().voxelNum.z; ++i) {
    projectionPtrs.push_back(ctImage.cspan().data() + i * projPixelSize);
  }

  GeometryCorrectionParams params;
  params.ballNum = ball_num;
  params.ballDis = ball_distance;
  params.thresholdPercent = threshold_percent;
  openpni::p2df pixelSize;
  pixelSize.x = pixel_size_u;
  pixelSize.y = pixel_size_v;

  auto result = GeometryCorrection(projectionPtrs.data(), ctImage.imageGeometry().voxelNum.z, proj_size_u, proj_size_v,
                                   pixelSize, params);
  openpni::autogen::json::GeoCorrParams geo_params;
  if (result.success) {
    std::cout << "Geometry correction completed successfully!" << std::endl;
    std::cout << "  SOD: " << result.SOD << " mm" << std::endl;
    std::cout << "  SDD: " << result.SDD << " mm" << std::endl;
    std::cout << "  OffsetU: " << result.offsetU << " mm" << std::endl;
    std::cout << "  OffsetV: " << result.offsetV << " mm" << std::endl;
    std::cout << "  Rotation: " << result.rotationAngle << " rad" << std::endl;

    geo_params.Geo_SOD = result.SOD;
    geo_params.Geo_SDD = result.SDD;
    geo_params.Geo_offsetU = result.offsetU;
    geo_params.Geo_offsetV = result.offsetV;
    geo_params.Geo_angle = result.rotationAngle;

    // 手动构造JSON对象
    json json_obj;
    json_obj["Geo_SOD"] = geo_params.Geo_SOD;
    json_obj["Geo_SDD"] = geo_params.Geo_SDD;
    json_obj["Geo_offsetU"] = geo_params.Geo_offsetU;
    json_obj["Geo_offsetV"] = geo_params.Geo_offsetV;
    json_obj["Geo_angle"] = geo_params.Geo_angle;

    // 输出为JSON文件
    std::string output_path = "config/GeoCorrParams.json";
    std::ofstream json_file(output_path);
    if (json_file.is_open()) {
      json_file << json_obj.dump(2) << std::endl;
      json_file.close();
      std::cout << "Geometry parameters saved to: " << output_path << std::endl;
    } else {
      std::cerr << "Failed to open file: " << output_path << std::endl;
    }
    return 0;
  } else {
    std::cerr << "Geometry correction failed!" << std::endl;
    return 1;
  }
}
