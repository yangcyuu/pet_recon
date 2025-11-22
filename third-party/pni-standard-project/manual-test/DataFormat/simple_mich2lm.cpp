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

void simple_mich2lm() {
#define DEBUGPATH 1
#ifdef DEBUGPATH
  std::filesystem::current_path("/home/ustc/Desktop/testBi_dynamicCase");
#endif
  std::cout << "Current working directory: " << std::filesystem::current_path() << std::endl;

  std::string in_mich = "/home/ustc/Desktop/testBi_dynamicCase/Data/source/coin_all.image3d";
  std::string out_lm = "/home/ustc/Desktop/testBi_dynamicCase/Data/source/lm.pni";
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
  auto converter = core::IndexConverter::create(mich);

  // 批次配置
  constexpr size_t batchSize_GB = 4;
  constexpr size_t batchSize_elements = (batchSize_GB * 1024ULL * 1024ULL * 1024ULL) / sizeof(float);
  const size_t totalElements = michInfo.getMichSize();

  std::cout << "Total elements: " << totalElements << " ("
            << (totalElements * sizeof(float) / (1024.0 * 1024.0 * 1024.0)) << " GB)" << std::endl;
  std::cout << "Batch size: " << batchSize_elements << " (" << batchSize_GB << " GB)" << std::endl;

  // 创建ListmodeFileOutput对象
  openpni::io::listmode::ListmodeFileOutput listmodeOutput;
  listmodeOutput.setBytes4CrystalIndex(openpni::io::listmode::CrystalIndexType::UINT32);
  listmodeOutput.setBytes4TimeValue1_2(openpni::io::listmode::TimeValue1_2Type::ZERO);
  listmodeOutput.open(out_lm);

  // 分批处理文件
  for (size_t offset = 0; offset < totalElements; offset += batchSize_elements) {
    size_t currentBatchSize = std::min(batchSize_elements, totalElements - offset);

    std::cout << "Processing batch: " << (offset / batchSize_elements + 1) << ", offset: " << offset
              << ", size: " << currentBatchSize << " ("
              << (currentBatchSize * sizeof(float) / (1024.0 * 1024.0 * 1024.0)) << " GB)" << std::endl;
    auto michBatch = read_from_file_batch<float>(in_mich, currentBatchSize, offset);

    // 当前批次的 listmode 数据
    std::vector<openpni::basic::Listmode_t> lmBatchData;

    // 处理当前批次
    for (size_t i = 0; i < currentBatchSize; ++i) {
      size_t globalLorIndex = offset + i; // 全局 LOR 索引
      if (michBatch[i] > 0) {
        const int count = static_cast<int>(std::ceil(michBatch[i]));

        // 预先计算坐标转换（避免重复计算）
        auto [cry1R, cry2R] = converter.getCrystalIDFromLORID(globalLorIndex);
        auto cry1U = converter.getUniformIDFromRectangleID(cry1R);
        auto cry2U = converter.getUniformIDFromRectangleID(cry2R);

        // 批量添加
        for (int j = 0; j < 1; ++j) {
          openpni::basic::Listmode_t lm;
          lm.globalCrystalIndex1 = converter.getFlatIdFromUniformID(cry1U);
          lm.globalCrystalIndex2 = converter.getFlatIdFromUniformID(cry2U);
          lm.time1_2pico = 0; // 时间差设为0
          lmBatchData.push_back(lm);
        }
      }
    }

    // 写入当前批次的数据
    if (!lmBatchData.empty()) {
      if (!listmodeOutput.appendSegment(lmBatchData.data(), lmBatchData.size(), 0, 0)) {
        std::cerr << "Error writing listmode batch " << (offset / batchSize_elements + 1) << std::endl;
        break;
      }
      std::cout << "Written " << lmBatchData.size() << " listmode events from batch "
                << (offset / batchSize_elements + 1) << std::endl;
    }

    // 清理当前批次数据，释放内存
    lmBatchData.clear();
    lmBatchData.shrink_to_fit();
  }

  std::cout << "Conversion completed successfully!" << std::endl;
}

int main() {
  simple_mich2lm();
  return 0;
}