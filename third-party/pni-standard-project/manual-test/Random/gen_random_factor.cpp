#include <filesystem>
#include <format>
#include <fstream>
#include <ranges>

#include "../public.hpp"
#include "include/experimental/node/MichRandom.hpp"
#include "include/experimental/tools/Parallel.hpp"
using namespace openpni::experimental;
/*
Manual Test for MichRandom Factor Generation
params:
  PolygonalSystem define
  RandomParams randParams
  Data File String
  Data File Offset
*/

int main() {
  std::cout << "\n=== Mich Random Factor Generation Test ===\n" << std::endl;
  //==============================================================CONFIG LOADING
  std::cout << "Current working directory: " << std::filesystem::current_path() << std::endl;

  std::string polygonDefineFilePath = "config/polygonSystemDefine.json";
  std::string randParamsFilePath = "config/randomParams.json";

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

  auto randParams = readFromJson<openpni::autogen::json::RandomParams>(randParamsFilePath);
  std::cout << "\n=== Configuration loaded successfully! ===\n" << std::endl;

  //==============================================================MAIN
  // 使用配置
  tools::cpu_threads().setThreadNumType(tools::MAX_THREAD).setScheduleType(tools::DYNAMIC).setScheduleNum(64);
  auto michInfo = core::MichInfoHub::create(mich);
  std::string fileIn = "Data/source/delay.pni";
  auto delayData = read_from_file<float>(fileIn, michInfo.getMichSize(), 6);
  openpni::experimental::node::MichRandom rand(mich);
  rand.setMinSectorDifference(randParams.minSectorDifference);
  rand.setRadialModuleNumS(randParams.radialModuleNumS);
  rand.setDelayMich(delayData.get());
  auto result = rand.dumpFactorsAsHMich();
  std::string fileOut = "Data/result/random_factor.pni";
  write_to_file<float>(fileOut, result.get(), michInfo.getMichSize());
  std::cout << "\n\033[92m✓ Random factors saved to " << fileOut << "\033[0m" << std::endl;
  std::cout << "\n=== Random factor generation completed successfully! ===\n" << std::endl;
  return 0;
}
