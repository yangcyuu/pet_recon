#include <cmath> // for std::ceil
#include <include/io/ListmodeIO.hpp>
#include <iostream>

#include "../public.hpp"
#include "include/basic/CudaPtr.hpp"
#include "include/basic/PetDataType.h"
#include "include/experimental/core/Image.hpp"
#include "include/experimental/core/Mich.hpp"
#include "include/experimental/example/EasyParallel.hpp"
#include "include/experimental/node/MichCrystal.hpp"
using namespace openpni::experimental;

struct BiDynamicLM {
  float cry1RectangleID1; // biDynamic use rectangleID to index crystals
  float cry2RectangleID2;
  float value; // the count,every event has instant value 1 in biDynamic
};

template <typename LMType>
void Lm2PNILm() {
  std::string in_BiDyamic_lm = "/home/ustc/Desktop/BiDynamic_1/Data/source/DerenzoR200F18_Ori_3.000000_CenterPos.coor";
  std::string out_pni_lm = "/home/ustc/Desktop/BiDynamic_1/Data/source/lm.pni";
  std::string polygonDefineFilePath = "/home/ustc/Desktop/BiDynamic_1/config/polygonSystemDefine.json";

  std::ifstream inputFile(in_BiDyamic_lm, std::ios::binary);
  if (!inputFile.is_open()) {
    std::cerr << "Error opening input file: " << in_BiDyamic_lm << std::endl;
    return;
  }

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
  auto converter = core::IndexConverter::create(mich);
  std::size_t batchSize_GB = 2;
  std::size_t batchSize_elements = (batchSize_GB * 1024ULL * 1024ULL * 1024ULL) / sizeof(LMType);
  std::size_t totalElements = std::filesystem::file_size(in_BiDyamic_lm) / sizeof(LMType);

  std::cout << "total file size: " << std::filesystem::file_size(in_BiDyamic_lm) << " bytes." << std::endl;
  std::cout << "Total elements: " << totalElements << " ("
            << (totalElements * sizeof(LMType) / (1024.0 * 1024.0 * 1024.0)) << " GB)" << std::endl;
  std::cout << "Batch size: " << batchSize_elements << " (" << batchSize_GB << " GB)" << std::endl;

  // 创建ListmodeFileOutput对象
  openpni::io::listmode::ListmodeFileOutput listmodeOutput;
  listmodeOutput.setBytes4CrystalIndex(openpni::io::listmode::CrystalIndexType::UINT32);
  listmodeOutput.setBytes4TimeValue1_2(openpni::io::listmode::TimeValue1_2Type::ZERO);
  listmodeOutput.open(out_pni_lm);

  for (size_t offset = 0; offset < totalElements; offset += batchSize_elements) {
    size_t currentBatch_elements = std::min(batchSize_elements, totalElements - offset);

    std::cout << "Processing batch: " << (offset / batchSize_elements + 1) << ", offset: " << offset
              << ", size: " << currentBatch_elements << " ("
              << (currentBatch_elements * sizeof(LMType) / (1024.0 * 1024.0 * 1024.0)) << " GB)" << std::endl;

    std::streampos seekPosition = offset * sizeof(LMType);
    inputFile.seekg(seekPosition, std::ios::beg);
    if (!inputFile.good()) {
      std::cerr << "Error seeking to position " << seekPosition << std::endl;
      break;
    }
    // 读取当前批次的数据
    std::vector<LMType> batch(currentBatch_elements);
    inputFile.read(reinterpret_cast<char *>(batch.data()), currentBatch_elements * sizeof(LMType));

    if (!inputFile.good() && !inputFile.eof()) {
      std::cerr << "Error reading batch data" << std::endl;
      break;
    }
    std::vector<openpni::basic::Listmode_t> lm;

    for (auto originLM : batch) {
      for (size_t count = 0; count < static_cast<size_t>(originLM.value); count++) {
        openpni::basic::Listmode_t pniData;
        auto cry1R = converter.getRectangleIdFromFlatId(static_cast<uint32_t>(originLM.cry1RectangleID1));
        auto cry2R = converter.getRectangleIdFromFlatId(static_cast<uint32_t>(originLM.cry2RectangleID2));
        auto cry1U = converter.getUniformIDFromRectangleID(cry1R);
        auto cry2U = converter.getUniformIDFromRectangleID(cry2R);
        pniData.globalCrystalIndex1 = converter.getFlatIdFromUniformID(cry1U);
        pniData.globalCrystalIndex2 = converter.getFlatIdFromUniformID(cry2U);
        pniData.time1_2pico = 0; // biDynamic时间差设为0
        lm.push_back(pniData);
      }
    }
    // 写入当前批次的数据
    if (!lm.empty()) {
      if (!listmodeOutput.appendSegment(lm.data(), lm.size(), 0, 0)) {
        std::cerr << "Error writing listmode batch " << (offset / batchSize_elements + 1) << std::endl;
        break;
      }
      std::cout << "Written " << lm.size() << " listmode events from batch " << (offset / batchSize_elements + 1)
                << std::endl;
    }

    // 清理当前批次数据，释放内存
    lm.clear();
    lm.shrink_to_fit();
  }
}

int main() {
  Lm2PNILm<BiDynamicLM>();
  return 0;
}
