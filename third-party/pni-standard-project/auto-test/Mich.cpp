#include "include/experimental/core/Mich.hpp"

#include <gtest/gtest.h>
#define TEST_SUITE_NAME openpni_experimental_core_Mich
#define test(name) TEST(TEST_SUITE_NAME, name)
using namespace openpni::experimental::core;

auto e180() {
  MichDefine mich;
  auto &polygon = mich.polygon;
  polygon.edges = 24;
  polygon.detectorPerEdge = 1;
  polygon.detectorLen = 0;
  polygon.radius = 106.5;
  polygon.angleOf1stPerp = 0;
  polygon.detectorRings = 2;
  polygon.ringDistance = 26.5 * 4 + 2;
  auto &detector = mich.detector;
  detector.blockNumU = 4;
  detector.blockNumV = 1;
  detector.blockSizeU = 26.5;
  detector.blockSizeV = 26.5;
  detector.crystalNumU = 13;
  detector.crystalNumV = 13;
  detector.crystalSizeU = 2.0;
  detector.crystalSizeV = 2.0;
  return mich;
}

test(
    小动物E180信息测试) {
  MichDefine mich = e180();

  auto michInfoHub = MichInfoHub::create(mich);
  EXPECT_EQ(michInfoHub.getCrystalNumOneRing(), 24 * 1 * 1 * 13);                  // 312
  EXPECT_EQ(michInfoHub.getRingNum(), 13 * 2 * 4);                                 // 104
  EXPECT_EQ(michInfoHub.getBinNum(), 311);                                         // 156
  EXPECT_EQ(michInfoHub.getViewNum(), 156);                                        // 96
  EXPECT_EQ(michInfoHub.getSliceNum(), 10816);                                     // 104
  EXPECT_EQ(michInfoHub.getLORNum(), size_t(311) * size_t(156) * size_t(10816));   // 524749056
  EXPECT_EQ(michInfoHub.getMichSize(), size_t(311) * size_t(156) * size_t(10816)); // 524749056
  EXPECT_EQ(michInfoHub.getPanelNum(), 24);                                        // 24
  EXPECT_EQ(michInfoHub.getBlockNumOneRing(), 24 * 1 * 1);                         // 24
  EXPECT_EQ(michInfoHub.getTotalCrystalNum(), 312 * 104);                          // 32448
  EXPECT_EQ(michInfoHub.getTotalBlockNum(), 24 * 1 * 1 * 4 * 2);                   // 192
  EXPECT_EQ(michInfoHub.getCrystalNumZInModule(), 4 * 13);                         // 52
  EXPECT_EQ(michInfoHub.getCrystalNumYInModule(), 1 * 13);                         // 13
  EXPECT_EQ(michInfoHub.getCrystalNumInModule(), 52 * 13);                         // 676
  EXPECT_EQ(michInfoHub.getCrystalNumInModule(), 4 * 13 * 1 * 13);                 // 676
  EXPECT_EQ(michInfoHub.getCrystalNumZInPanel(), 8 * 13);                          // 104
  EXPECT_EQ(michInfoHub.getCrystalNumYInPanel(), 1 * 13);                          // 13
  EXPECT_EQ(michInfoHub.getCrystalNumInPanel(), 8 * 1 * 169);                      // 1352
  EXPECT_EQ(michInfoHub.getBlockNumZInPanel(), 4 * 1 * 2);                         // 8
  EXPECT_EQ(michInfoHub.getBlockNumYInPanel(), 1);                                 // 1
  EXPECT_EQ(michInfoHub.getBlockNumInPanel(), 8 * 1);                              // 8
}

test(
    小动物E180转换测试) {
  MichDefine mich = e180();

  auto indexConverter = IndexConverter::create(mich);
  EXPECT_EQ(indexConverter.getBVSFromLOR(0), (Vector<uint32_t, 3>{0, 0, 0}));
  EXPECT_EQ(indexConverter.getBVSFromLOR(311 * 156 * 10816 - 1), (Vector<uint32_t, 3>{310, 155, 10815}));
  EXPECT_EQ(indexConverter.getLORFromBVS(0, 0, 0), 0);
  EXPECT_EQ(indexConverter.getLORFromBVS(310, 155, 10815), 311 * 156 * 10816 - 1);
  EXPECT_EQ(indexConverter.getCrystalInRingFromBinView(0, 0), (Vector<uint32_t, 2>{0, 1}));
  EXPECT_EQ(indexConverter.getCrystalInRingFromBinView(310, 155), (Vector<uint32_t, 2>{0, 311}));
  EXPECT_EQ(indexConverter.getRing1Ring2FromSlice(0), (Vector<uint32_t, 2>{0, 0}));
  EXPECT_EQ(indexConverter.getRing1Ring2FromSlice(10815), (Vector<uint32_t, 2>{103, 103}));
  EXPECT_EQ(indexConverter.getSliceFromRing1Ring2(0, 0), 0);
  EXPECT_EQ(indexConverter.getSliceFromRing1Ring2(103, 103), 10815);
  {
    auto [rid1, rid2] = indexConverter.getCrystalIDFromLORID(0);
    EXPECT_EQ(rid1.idInRing, 1);
    EXPECT_EQ(rid1.ringID, 0);
    EXPECT_EQ(rid2.idInRing, 0);
    EXPECT_EQ(rid2.ringID, 0);
  }
  {
    auto [rid1, rid2] = indexConverter.getCrystalIDFromLORID(311 * 156 * 10816 - 1);
    EXPECT_EQ(rid1.idInRing, 311);
    EXPECT_EQ(rid1.ringID, 103);
    EXPECT_EQ(rid2.idInRing, 0);
    EXPECT_EQ(rid2.ringID, 103);
  }
  EXPECT_EQ(indexConverter.getLORIDFromRectangleID({0, 1}, {0, 0}), 0);
  EXPECT_EQ(indexConverter.getLORIDFromRectangleID({103, 311}, {103, 0}), 311 * 156 * 10816 - 1);
  EXPECT_EQ(indexConverter.getBlockIdInRingFromCrystalId({0, 0}), 0);
  EXPECT_EQ(indexConverter.getBlockIdInRingFromCrystalId({0, 12}), 0);
  EXPECT_EQ(indexConverter.getBlockIdInRingFromCrystalId({0, 13}), 1);
  EXPECT_EQ(indexConverter.getBlockIdInRingFromCrystalId({0, 51}), 3);
  EXPECT_EQ(indexConverter.getBlockIdInRingFromCrystalId({0, 52}), 4);
  EXPECT_EQ(indexConverter.getBlockIdInRingFromCrystalId({0, 311}), 23);
  EXPECT_TRUE(indexConverter.isGoodPairMinSector(0, 12, 4));
  EXPECT_FALSE(indexConverter.isGoodPairMinSector(0, 3, 4));
  EXPECT_FALSE(indexConverter.isGoodPairMinSector(0, 0, 4));
  EXPECT_TRUE(indexConverter.isGoodPairMinSector(0, 5, 4));
  EXPECT_FALSE(indexConverter.isGoodPairMinSector(0, 23, 4));
  {
    auto rid = indexConverter.getRectangleIDFromUniformID({0, 0, 0});
    EXPECT_EQ(rid.ringID, 0);
    EXPECT_EQ(rid.idInRing, 12);
  }
  {
    auto rid = indexConverter.getRectangleIDFromUniformID({47, 51, 12});
    EXPECT_EQ(rid.ringID, 103);
    EXPECT_EQ(rid.idInRing, 311 - 12);
  }
  {
    auto uid = indexConverter.getUniformIDFromRectangleID({0, 0});
    EXPECT_EQ(uid.detectorID, 0);
    EXPECT_EQ(uid.crystalU, 0);
    EXPECT_EQ(uid.crystalV, 12);
  }
  {
    auto uid = indexConverter.getUniformIDFromRectangleID({103, 311});
    EXPECT_EQ(uid.detectorID, 47);
    EXPECT_EQ(uid.crystalU, 51);
    EXPECT_EQ(uid.crystalV, 0);
  }
  EXPECT_EQ(indexConverter.getFlatIdFromRectangleId({0, 0}), 0);
  EXPECT_EQ(indexConverter.getFlatIdFromRectangleId({0, 311}), 311);
  EXPECT_EQ(indexConverter.getFlatIdFromRectangleId({1, 0}), 312);
  EXPECT_EQ(indexConverter.getFlatIdFromRectangleId({103, 311}), 312 * 104 - 1);
  EXPECT_EQ(indexConverter.getFlatIdFromUniformID({0, 0, 0}), 0);
  EXPECT_EQ(indexConverter.getFlatIdFromUniformID({47, 51, 12}), 312 * 104 - 1);
  {
    auto rid = indexConverter.getRectangleIdFromFlatId(0);
    EXPECT_EQ(rid.ringID, 0);
    EXPECT_EQ(rid.idInRing, 0);
  }
  {
    auto rid = indexConverter.getRectangleIdFromFlatId(312 * 104 - 1);
    EXPECT_EQ(rid.ringID, 103);
    EXPECT_EQ(rid.idInRing, 311);
  }
  {
    auto rid = indexConverter.getRectangleIdFromFlatId(311);
    EXPECT_EQ(rid.ringID, 0);
    EXPECT_EQ(rid.idInRing, 311);
  }
  {
    auto rid = indexConverter.getRectangleIdFromFlatId(312);
    EXPECT_EQ(rid.ringID, 1);
    EXPECT_EQ(rid.idInRing, 0);
  }
  {
    auto rid = indexConverter.getRectangleIdFromFlatId(311 + 103 * 312);
    EXPECT_EQ(rid.ringID, 103);
    EXPECT_EQ(rid.idInRing, 311);
  }
  {
    auto rid = indexConverter.getRectangleIdFromFlatId(0);
    EXPECT_EQ(rid.ringID, 0);
    EXPECT_EQ(rid.idInRing, 0);
  }
  {
    auto rid = indexConverter.getRectangleIdFromFlatId(312 * 104 - 1);
    EXPECT_EQ(rid.ringID, 103);
    EXPECT_EQ(rid.idInRing, 311);
  }
  {
    auto rid = indexConverter.getRectangleIdFromFlatId(311);
    EXPECT_EQ(rid.ringID, 0);
    EXPECT_EQ(rid.idInRing, 311);
  }
  {
    auto rid = indexConverter.getRectangleIdFromFlatId(312);
    EXPECT_EQ(rid.ringID, 1);
    EXPECT_EQ(rid.idInRing, 0);
  }
  {
    auto rid = indexConverter.getRectangleIdFromFlatId(311 + 103 * 312);
    EXPECT_EQ(rid.ringID, 103);
    EXPECT_EQ(rid.idInRing, 311);
  }
  EXPECT_EQ(indexConverter.getRing1Ring2FromSlice(0), (Vector<uint32_t, 2>{0, 0}));
  EXPECT_EQ(indexConverter.getRing1Ring2FromSlice(10816 - 1), (Vector<uint32_t, 2>{104 - 1, 104 - 1}));
}