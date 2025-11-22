#include <iostream>

#include "../public.hpp"
#include "include/experimental/core/Image.hpp"
#include "include/experimental/node/MichCrystal.hpp"

using namespace openpni::experimental;

int main() {
  std::filesystem::current_path("/home/ustc/Desktop/BiDynamic_2");
  std::cout << "Current working directory: " << std::filesystem::current_path() << std::endl;

  std::string polygonDefineFilePath = "config/polygonSystemDefine.json";
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
  auto totalCrystals = michInfo.getTotalCrystalNum();
  auto totalBins = michInfo.getBinNum();
  auto totalViews = michInfo.getViewNum();
  auto totalSlices = michInfo.getSliceNum();
  auto totalLors = michInfo.getLORNum();
  auto totalRings = michInfo.getRingNum();
  auto totalBlockRings = michInfo.getBlockRingNum();
  auto totalBlockNums = michInfo.getTotalBlockNum();
  std::cout << "Total Crystals: " << totalCrystals << std::endl;
  std::cout << "Total Bins: " << totalBins << std::endl;
  std::cout << "Total Views: " << totalViews << std::endl;
  std::cout << "Total Slices: " << totalSlices << std::endl;
  std::cout << "Total LORs: " << totalLors << std::endl;
  std::cout << "Total Rings: " << totalRings << std::endl;
  std::cout << "Total Block Rings: " << totalBlockRings << std::endl;
  std::cout << "Total Block Nums: " << totalBlockNums << std::endl;

  auto michCrystal = node::MichCrystal(mich);
  auto CrystalGeom = michCrystal.dumpCrystalsUniformLayout();
  auto crystalGeoRectang = michCrystal.dumpCrystalsRectangleLayout();
  std::vector<p3df> crystalPos;
  std::ranges::transform(CrystalGeom, std::back_inserter(crystalPos), [](const auto &crystal) { return crystal.O; });

  // 将晶体坐标打印到800*800*800的图内
  auto imgGrids = core::Grids<3, float>::create_by_spacing_size(core::Vector<float, 3>::create(0.5f, 0.5f, 0.5f),
                                                                core::Vector<int64_t, 3>::create(1000, 1000, 1000));
  // 遍历晶体坐标，放入图像
  std::vector<float> imgData(imgGrids.totalSize(), 0.0f);
  for (const auto &pos : crystalPos) {
    // 计算坐标对应的体素索引
    auto indexF = (pos.to<float>() - imgGrids.origin) / imgGrids.spacing;
    auto index = core::Vector<int64_t, 3>::create(static_cast<int64_t>(std::round(indexF[0])),
                                                  static_cast<int64_t>(std::round(indexF[1])),
                                                  static_cast<int64_t>(std::round(indexF[2])));
    // std::cout << "Index: (" << index[0] << ", " << index[1] << ", " << index[2] << ")" << std::endl;

    // 判断索引是否在图像范围内
    size_t voxelIndex =
        index[0] * imgGrids.size.dimSize[1] * imgGrids.size.dimSize[2] + index[1] * imgGrids.size.dimSize[2] + index[2];
    imgData[voxelIndex] = 1.0f; // 设置体素值为1.0
  }
  // save
  std::ofstream outFile("systemCrystalPos.bin", std::ios::binary);
  outFile.write(reinterpret_cast<char *>(imgData.data()), imgData.size() * sizeof(float));
  outFile.close();

  // test cryid
  auto converter = core::IndexConverter::create(mich);
  for (auto lortest : std::views::iota(0u, 10u)) {
    auto [cry1R, cry2R] = converter.getCrystalIDFromLORID(lortest);
    auto cry1FR = converter.getFlatIdFromRectangleId(cry1R);
    auto cry2FR = converter.getFlatIdFromRectangleId(cry2R);
    std::cout << "LOR ID: " << lortest << " => Crystal1 RectangleID: (Ring " << cry1R.ringID << ", IDInRing "
              << cry1R.idInRing << "), FlatID: " << cry1FR << " | Crystal2 RectangleID: (Ring " << cry2R.ringID
              << ", IDInRing " << cry2R.idInRing << "), FlatID: " << cry2FR << std::endl;
    // only print FlatRectangleID
    // print cry1R cry2R position
    auto cryPos1 = crystalGeoRectang[cry1FR].O;
    auto cryPos2 = crystalGeoRectang[cry2FR].O;
    std::cout << "    => Crystal1 Position: (" << cryPos1[0] << ", " << cryPos1[1] << ", " << cryPos1[2] << ")"
              << " | Crystal2 Position: (" << cryPos2[0] << ", " << cryPos2[1] << ", " << cryPos2[2] << ")"
              << std::endl;
  }

  size_t testLORID = 3808;
  auto [cry1R, cry2R] = converter.getCrystalIDFromLORID(testLORID);
  auto cry1FR = converter.getFlatIdFromRectangleId(cry1R);
  auto cry2FR = converter.getFlatIdFromRectangleId(cry2R);
  auto cry1U = converter.getUniformIDFromRectangleID(cry1R);
  auto cry2U = converter.getUniformIDFromRectangleID(cry2R);
  auto cry1FU = converter.getFlatIdFromUniformID(cry1U);
  auto cry2FU = converter.getFlatIdFromUniformID(cry2U);
  std::cout << "LOR ID: " << testLORID << " => Crystal1 RectangleID: (Ring " << cry1R.ringID << ", IDInRing "
            << cry1R.idInRing << "), FlatID: " << cry1FR << " | Crystal2 RectangleID: (Ring " << cry2R.ringID
            << ", IDInRing " << cry2R.idInRing << "), FlatID: " << cry2FR << std::endl;
  std::cout << "    => Crystal1 UniformID: (Detector " << cry1U.detectorID << ", U " << cry1U.crystalU << ", V "
            << cry1U.crystalV << "), FlatID: " << cry1FU << " | Crystal2 UniformID: (Detector " << cry2U.detectorID
            << ", U " << cry2U.crystalU << ", V " << cry2U.crystalV << "), FlatID: " << cry2FU << std::endl;
  // test
  auto cry3U = converter.getUniformIdFromFlatId(14040);
  auto cry3R = converter.getRectangleIDFromUniformID(cry3U);
  auto cry4U = converter.getUniformIdFromFlatId(11420);
  auto cry4R = converter.getRectangleIDFromUniformID(cry4U);
  auto lorReal = converter.getLORIDFromRectangleID(cry3R, cry4R);
  std::cout << "real LOR ID from Crystal3 UniformID FlatID 14040 and Crystal4 UniformID FlatID 11420: " << lorReal
            << std::endl;

  return 0;
}