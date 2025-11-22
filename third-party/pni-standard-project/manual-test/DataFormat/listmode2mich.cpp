#include <cmath> // for std::ceil
#include <iostream>

#include "../public.hpp"
#include "include/basic/CudaPtr.hpp"
#include "include/basic/PetDataType.h"
#include "include/experimental/core/Image.hpp"
#include "include/experimental/core/Mich.hpp"
#include "include/experimental/example/EasyParallel.hpp"
#include "include/experimental/node/MichCrystal.hpp"
#include "include/io/IO.hpp"
#include "include/io/ListmodeIO.hpp"
#include "include/misc/ListmodeBuffer.hpp"
using namespace openpni::experimental;

void PNILM2Mich() {

  std::string listmode_path = "/media/cmx/K1/v4_coin/data/save/clinic_algo_data/cylinder/pni_delay_lm.pni";
  std::string out_mich = "/media/cmx/K1/v4_coin/data/LGX_test/mich/delay.mich";

  std::string polygonDefineFilePath = "/media/cmx/K1/v4_coin/data/LGX_test/930define.json";

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
  auto converter = core::IndexConverter::create(mich);

  // read listmodeFile
  openpni::io::ListmodeFileInput listmodeFile;
  listmodeFile.open(listmode_path);
  auto selectedListmodeSegments = openpni::io::selectSegments(listmodeFile);
  auto totalEvents = std::accumulate(selectedListmodeSegments.begin(), selectedListmodeSegments.end(), 0ull,
                                     [](auto a, auto b) { return a + b.dataIndexEnd - b.dataIndexBegin; });
  PNI_DEBUG(std::format("Listmode file opened, reading segments... Total events: {}\n", totalEvents));

  // Prepare listmode buffer for chunked reading
  openpni::misc::ListmodeBuffer listmodeBuffer;

  std::vector<float> michData(michInfo.getMichSize());
  auto GBSize = [](unsigned long long size) -> uint64_t { return size * 1024 * 1024 * 1024; };

  listmodeBuffer.setBufferSize(GBSize(4) / sizeof(openpni::basic::Listmode_t))
      .callWhenFlush([&](const openpni::basic::Listmode_t *__data, std::size_t __count) {
        for (auto i : std::views::iota(0ull, __count)) {
          const auto &lm = __data[i];
          auto globalCrystalID1 = lm.globalCrystalIndex1;
          auto globalCrystalID2 = lm.globalCrystalIndex2;
          // if( globalCrystalID1 >= michInfo.getTotalCrystalNum() ||
          //     globalCrystalID2 >= michInfo.getTotalCrystalNum()) {
          //   continue;
          // }
          auto u1 = converter.getUniformIdFromFlatId(globalCrystalID1);
          auto u2 = converter.getUniformIdFromFlatId(globalCrystalID2);
          auto r1 = converter.getRectangleIDFromUniformID(u1);
          auto r2 = converter.getRectangleIDFromUniformID(u2);
          auto lor = converter.getLORIDFromRectangleID(r1, r2);
          michData[lor] += 1.0f;
        }
      })
      .append(listmodeFile, openpni::io::selectSegments(listmodeFile))
      .flush();

  // Save mich data
  {
    std::ofstream ofs(out_mich, std::ios::binary);
    ofs.write(reinterpret_cast<const char *>(michData.data()), michData.size() * sizeof(float));
    ofs.close();
    PNI_DEBUG(std::format("Mich data saved to {}\n", out_mich));
  }
}

int main() {
  PNILM2Mich();
  return 0;
}