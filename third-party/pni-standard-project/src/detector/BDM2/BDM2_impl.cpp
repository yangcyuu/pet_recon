#include "BDM2_impl.hpp"

#include <iostream>
#include <random>
#include <ranges>

#include "include/misc/HeaderWithSizeReserved.hpp"
#include "include/misc/Platform-Independent.hpp"
#include "kernel.hpp"
#include "src/autogen/autogen_xml.hpp"
#include "src/common/FileFormat.hpp"
namespace openpni::device::bdm2 {
using openpni::misc::flt_max;
using CaliFileHeader =
    misc::HeaderWithSizeReserved<openpni::autogen::GeneralFileHeader, openpni::autogen::GeneralFileHeaderSize>;
KMeansInitialLine_t getAutoInitialX(
    const CountMap_t &__countMap) {
  auto random = [] {
    thread_local static std::mt19937 rng;
    return rng();
  };

  std::array<unsigned, CRYSTAL_RAW_POSITION_RANGE> countLine;
  countLine.fill(0);
  KMeansInitialLine_t result;
  for (int x = 0; x < CRYSTAL_RAW_POSITION_RANGE; x++) {
    for (int y = 0; y < CRYSTAL_RAW_POSITION_RANGE; y++)
      countLine[x] += __countMap[x + y * CRYSTAL_RAW_POSITION_RANGE];
  }
  const unsigned maxSumX = *std::max_element(countLine.begin(), countLine.end());

  unsigned currentIndex = 0;
  for (int i = 0; i < countLine.size(); i++) {
    if (countLine[i] < maxSumX / 3)
      continue;
    if (i <= 1)
      continue;
    if (i >= 254)
      continue;
    if (countLine[i] > countLine[i - 1] && countLine[i] > countLine[i - 2] && countLine[i] > countLine[i + 1] &&
        countLine[i] > countLine[i + 2] && currentIndex < CRYSTAL_LINE)
      result[currentIndex++] = i;
  }
  for (; currentIndex < CRYSTAL_LINE; currentIndex++)
    result[currentIndex] = random() % CRYSTAL_LINE;
  return result;
}
KMeansInitialLine_t getAutoInitialY(
    const CountMap_t &__countMap) {
  std::array<unsigned, CRYSTAL_RAW_POSITION_RANGE> countLine;
  countLine.fill(0);
  KMeansInitialLine_t result;
  for (int y = 0; y < CRYSTAL_RAW_POSITION_RANGE; y++) {
    for (int x = 0; x < CRYSTAL_RAW_POSITION_RANGE; x++)
      countLine[y] += __countMap[x + y * CRYSTAL_RAW_POSITION_RANGE];
  }

  const unsigned maxSumY = *std::max_element(countLine.begin(), countLine.end());
  unsigned currentIndex = 0;
  for (int i = 0; i < countLine.size(); i++) {
    if (countLine[i] < maxSumY / 3)
      continue;
    if (i <= 1)
      continue;
    if (i >= 254)
      continue;
    if (countLine[i] > countLine[i - 1] && countLine[i] > countLine[i - 2] && countLine[i] > countLine[i + 1] &&
        countLine[i] > countLine[i + 2] && currentIndex < CRYSTAL_LINE)
      result[currentIndex++] = i;
  }
  for (; currentIndex < CRYSTAL_LINE; currentIndex++)
    result[currentIndex] = random() % CRYSTAL_LINE;
  return result;
}

template <unsigned CountMapSize, unsigned CenterPointSize, unsigned Iteration, typename CenterPointIndexType,
          typename ResultMapValueType>
void KMeans(
    const std::array<unsigned, CountMapSize * CountMapSize> &__countMap,
    const std::array<basic::Vec2<CenterPointIndexType>, CenterPointSize * CenterPointSize> &__initialCenterPoints,
    std::array<basic::Vec2<CenterPointIndexType>, CenterPointSize * CenterPointSize> &__centrePoints_out,
    std::array<ResultMapValueType, CountMapSize * CountMapSize> &__mapPixel2CenterPointIndex_out) {
  __centrePoints_out = __initialCenterPoints;

  for (int i = 0; i < Iteration; i++) {
#pragma omp parallel for // Calculate the nearest center point for each pixel
    for (unsigned pixelIndex = 0; pixelIndex < __countMap.size(); pixelIndex++) {
      const int x = pixelIndex % CountMapSize;
      const int y = pixelIndex / CountMapSize;
      __mapPixel2CenterPointIndex_out[pixelIndex] = 0;
      float nearestDistance = flt_max;
      for (unsigned centerIndex = 0; centerIndex < __centrePoints_out.size(); centerIndex++) {
        const auto dist = (x - __centrePoints_out[centerIndex].x) * (x - __centrePoints_out[centerIndex].x) +
                          (y - __centrePoints_out[centerIndex].y) * (y - __centrePoints_out[centerIndex].y);
        if (dist < nearestDistance) {
          nearestDistance = dist;
          __mapPixel2CenterPointIndex_out[pixelIndex] = centerIndex;
        }
      }
    }
    if (i == Iteration - 1)
      break;

    // Update the new Center
    std::array<basic::Vec2<double>, CenterPointSize * CenterPointSize> newCenterPoints;
    newCenterPoints.fill(basic::make_vec2<double>(0, 0));
    std::array<float, CenterPointSize * CenterPointSize> centerWeight;
    centerWeight.fill(0.0f);
    for (const auto rawIndex : std::views::iota(0ull, CountMapSize * CountMapSize)) {
      const auto pIndex = __mapPixel2CenterPointIndex_out[rawIndex];
      newCenterPoints[pIndex] +=
          basic::make_vec2<double>(rawIndex % CountMapSize, rawIndex / CountMapSize) * __countMap[rawIndex];
      centerWeight[pIndex] += __countMap[rawIndex];
    }

    for (unsigned pIndex = 0; pIndex < newCenterPoints.size(); pIndex++) {
      __centrePoints_out[pIndex] = (__centrePoints_out[pIndex] + newCenterPoints[pIndex] / centerWeight[pIndex]) / 2.;
    }
  }
}

template <int ConvolutionKernelHalfSize, typename EnergyCountMap, typename PositionMap>
void generateEnergyCountMap(
    EnergyCountMap &__outCountMap, const std::array<double, ConvolutionKernelHalfSize * 2 + 1> &__kernel,
    const std::vector<TempFrameV2> &__tempFrames, const PositionMap &__positionMap,
    const std::array<EnergyProfileCut_t, CRYSTAL_NUM_ONE_BLOCK> &__energyProfileCut) {
  static_assert(ConvolutionKernelHalfSize > 0, "ConvolutionKernelHalfSize must be greater than 0");
  static_assert(EnergyCountMap().size() == CRYSTAL_NUM_ONE_BLOCK,
                "EnergyCountMap must have size equal to CRYSTAL_NUM_ONE_BLOCK");
  constexpr auto countmapResolution = 1024; // Resolution of the energy count map
  // clear the output count map
  for (auto &c : __outCountMap)
    c.fill(0);
  // Count the energy map
  for (const auto &frame : __tempFrames) {
    const auto &crystalIndex = __positionMap[frame.x + frame.y * 256];
    const auto &energy = frame.energy;
    const auto &valueCutMin = __energyProfileCut[crystalIndex][0];
    const auto &valueCutMax = __energyProfileCut[crystalIndex][1];
    const int energyCountMapIndex = (energy - valueCutMin) / (valueCutMax - valueCutMin) * countmapResolution;
    if (energyCountMapIndex < 0 || energyCountMapIndex >= countmapResolution)
      continue;
    __outCountMap[crystalIndex][energyCountMapIndex]++;
  }
  // Do convolution for energy map
  for (auto &c : __outCountMap) {
    auto tempEnergyMap = c;
    for (int i = 0; i < c.size(); i++) {
      double sumCoef = 0;
      for (int j = -ConvolutionKernelHalfSize; j <= ConvolutionKernelHalfSize; j++) {
        if (i + j < 0 || i + j >= c.size())
          continue;
        tempEnergyMap[i] += c[i + j] * __kernel[j + ConvolutionKernelHalfSize];
        sumCoef += __kernel[j + ConvolutionKernelHalfSize];
      }
      if (sumCoef > 0)
        tempEnergyMap[i] /= sumCoef;
    }
    c = tempEnergyMap;
  }
}
} // namespace openpni::device::bdm2

namespace openpni::device::bdm2 {
void BDM2Calibrater_impl::loadCaliTable(
    const std::string &filename) // version == 1
{
  // 从文件中读取二进制数据
  auto stream = openpni::autogen::readBinaryFile(filename);
  if (!stream)
    throw openpni::exceptions::file_cannot_access();

  CaliFileHeader header;
  stream->seekg(0, std::ios_base::beg);
  stream->read(reinterpret_cast<char *>(&header), sizeof(header));
  if (!stream->good())
    throw openpni::exceptions::file_format_incorrect();
  // Skip sizeof(CaliFileHeader) bytes from the beginning of the stream
  auto stream_skip_header = openpni::autogen::subStream(*stream, sizeof(CaliFileHeader), stream->str().size());

#define try_assert(condition)                                                                                          \
  if (!(condition)) {                                                                                                  \
    throw openpni::exceptions::file_format_incorrect();                                                                \
  }

  try {
    if (header.header.version == 1) {
      auto caliFile =
          openpni::autogen::binary::struct_cast<openpni::autogen::binary::BDM2CaliFileV1>(stream_skip_header);
      m_h_caliTable = std::make_unique<CalibrationTable>();
      m_h_caliTable->energyCoef = std::make_unique<std::array<EnergyCoefs_t, BLOCK_NUM>>();
      m_h_caliTable->positionTable = std::make_unique<std::array<PositionTable_t, BLOCK_NUM>>();

      m_calibrationResult = std::make_unique<CalibrationResult>();
      m_calibrationResult->energyCoef = std::make_unique<std::array<EnergyCoefs_t, BLOCK_NUM>>();
      m_calibrationResult->positionTable = std::make_unique<std::array<PositionTable_t, BLOCK_NUM>>();
      m_calibrationResult->centerPosition = std::make_unique<std::array<CenterPosition_t, BLOCK_NUM>>();
      m_calibrationResult->energyProfile =
          std::make_unique<std::array<std::array<EnergyProfile_t, CRYSTAL_NUM_ONE_BLOCK>, BLOCK_NUM>>();
      m_calibrationResult->energyProfileCut =
          std::make_unique<std::array<std::array<EnergyProfileCut_t, CRYSTAL_NUM_ONE_BLOCK>, BLOCK_NUM>>();
      m_calibrationResult->countMap = std::make_unique<std::array<CountMap_t, BLOCK_NUM>>();

      try_assert(caliFile.EnergyCoefs.size() == BLOCK_NUM * CRYSTAL_NUM_ONE_BLOCK);
      for (int i = 0; i < BLOCK_NUM; ++i) {
        std::copy(caliFile.EnergyCoefs.begin() + i * CRYSTAL_NUM_ONE_BLOCK,
                  caliFile.EnergyCoefs.begin() + (i + 1) * CRYSTAL_NUM_ONE_BLOCK,
                  (*m_h_caliTable->energyCoef)[i].begin());
        std::copy(caliFile.EnergyCoefs.begin() + i * CRYSTAL_NUM_ONE_BLOCK,
                  caliFile.EnergyCoefs.begin() + (i + 1) * CRYSTAL_NUM_ONE_BLOCK,
                  (*m_calibrationResult->energyCoef)[i].begin());
      }

      try_assert(caliFile.PositionTable.size() == BLOCK_NUM * 256 * 256);
      for (int i = 0; i < BLOCK_NUM; ++i) {
        std::copy(caliFile.PositionTable.begin() + i * 256 * 256, caliFile.PositionTable.begin() + (i + 1) * 256 * 256,
                  (*m_h_caliTable->positionTable)[i].begin());
        std::copy(caliFile.PositionTable.begin() + i * 256 * 256, caliFile.PositionTable.begin() + (i + 1) * 256 * 256,
                  (*m_calibrationResult->positionTable)[i].begin());
      }

      try_assert(caliFile.CenterPosition.size() == BLOCK_NUM * CRYSTAL_NUM_ONE_BLOCK);
      m_centerPosition = std::make_unique<std::array<CenterPosition_t, BLOCK_NUM>>();
      for (int i = 0; i < BLOCK_NUM; ++i) {
        for (int j = 0; j < CRYSTAL_NUM_ONE_BLOCK; ++j) {
          (*m_centerPosition)[i][j].x = caliFile.CenterPosition[i * CRYSTAL_NUM_ONE_BLOCK + j].X;
          (*m_centerPosition)[i][j].y = caliFile.CenterPosition[i * CRYSTAL_NUM_ONE_BLOCK + j].Y;
          (*m_calibrationResult->centerPosition)[i][j].x = caliFile.CenterPosition[i * CRYSTAL_NUM_ONE_BLOCK + j].X;
          (*m_calibrationResult->centerPosition)[i][j].y = caliFile.CenterPosition[i * CRYSTAL_NUM_ONE_BLOCK + j].Y;
        }
      }

      try_assert(caliFile.EnergyProfile.size() == BLOCK_NUM * CRYSTAL_NUM_ONE_BLOCK * 1024);
      for (int i = 0; i < BLOCK_NUM; ++i) {
        for (int j = 0; j < CRYSTAL_NUM_ONE_BLOCK; ++j) {
          std::copy(caliFile.EnergyProfile.begin() + (i * CRYSTAL_NUM_ONE_BLOCK + j) * 1024,
                    caliFile.EnergyProfile.begin() + (i * CRYSTAL_NUM_ONE_BLOCK + j + 1) * 1024,
                    (*m_calibrationResult->energyProfile)[i][j].begin());
        }
      }

      try_assert(caliFile.EnergyProfileCut.size() == BLOCK_NUM * CRYSTAL_NUM_ONE_BLOCK * 2);
      for (int i = 0; i < BLOCK_NUM; ++i) {
        for (int j = 0; j < CRYSTAL_NUM_ONE_BLOCK; ++j) {
          (*m_calibrationResult->energyProfileCut)[i][j][0] =
              caliFile.EnergyProfileCut[(i * CRYSTAL_NUM_ONE_BLOCK + j) * 2];
          (*m_calibrationResult->energyProfileCut)[i][j][1] =
              caliFile.EnergyProfileCut[(i * CRYSTAL_NUM_ONE_BLOCK + j) * 2 + 1];
        }
      }

      try_assert(caliFile.CountMap.size() == BLOCK_NUM * 256 * 256);
      m_countMap = std::make_unique<std::array<CountMap_t, BLOCK_NUM>>();
      for (int i = 0; i < BLOCK_NUM; ++i) {
        std::copy(caliFile.CountMap.begin() + i * 256 * 256, caliFile.CountMap.begin() + (i + 1) * 256 * 256,
                  (*m_countMap)[i].begin());
        std::copy(caliFile.CountMap.begin() + i * 256 * 256, caliFile.CountMap.begin() + (i + 1) * 256 * 256,
                  (*m_calibrationResult->countMap)[i].begin());
      }

      // for (const auto &block : *m_calibrationResult->positionTable)
      // {
      //     for (const auto index : std::ranges::views::iota(0ull, 256 * 256ull))
      //     {
      //         printf("PositionTable[%lld][%lld] = %d\n", index / 256, index % 256,
      //         block[index]); if (block[index] >= CRYSTAL_NUM_ONE_BLOCK)
      //         {
      //             std::cerr << "Error: PositionTable index out of range: " <<
      //             block[index] << std::endl; exit(1);
      //         }
      //     }
      // }
      isCalibrationLoaded = true;
      return;
    }
  }

  catch (const std::exception &e) {
    throw openpni::exceptions::file_format_incorrect();
  }
  throw openpni::exceptions::file_unknown_version();
}

void BDM2Calibrater_impl::saveCaliTable(
    const std::string &filename) {
  // 检查 m_calibrationResult 是否已初始化
  if (!m_calibrationResult)
    throw std::runtime_error("CalibrationResult is not initialized.");

  // 将 CalibrationResult 转换为 BDM2CaliFileV1 格式
  openpni::autogen::binary::BDM2CaliFileV1 caliFile;
  caliFile.EnergyCoefs.reserve(BLOCK_NUM * CRYSTAL_NUM_ONE_BLOCK);
  caliFile.PositionTable.reserve(BLOCK_NUM * 256 * 256);
  caliFile.EnergyProfile.reserve(BLOCK_NUM * CRYSTAL_NUM_ONE_BLOCK * 1024);
  caliFile.CenterPosition.reserve(BLOCK_NUM * CRYSTAL_NUM_ONE_BLOCK);
  caliFile.CountMap.reserve(BLOCK_NUM * 256 * 256);
  caliFile.EnergyProfileCut.reserve(BLOCK_NUM * CRYSTAL_NUM_ONE_BLOCK * 2);

  for (const auto &block : *m_calibrationResult->energyCoef)
    caliFile.EnergyCoefs.insert(caliFile.EnergyCoefs.end(), block.begin(), block.end());

  for (const auto &block : *m_calibrationResult->positionTable)
    caliFile.PositionTable.insert(caliFile.PositionTable.end(), block.begin(), block.end());

  for (const auto &block : *m_calibrationResult->energyProfile)
    for (const auto &profile : block)
      caliFile.EnergyProfile.insert(caliFile.EnergyProfile.end(), profile.begin(), profile.end());

  for (const auto &block : *m_calibrationResult->centerPosition)
    for (const auto &point : block) {
      openpni::autogen::binary::UChar2D point2D;
      point2D.X = point.x;
      point2D.Y = point.y;
      caliFile.CenterPosition.emplace_back(point2D);
    }
  for (const auto &block : *m_calibrationResult->energyProfileCut)
    for (const auto &cut : block) {
      float cutValue = cut[0];
      float cutValue2 = cut[1];
      caliFile.EnergyProfileCut.push_back(cutValue);
      caliFile.EnergyProfileCut.push_back(cutValue2);
    }
  for (const auto &block : *m_calibrationResult->countMap)
    for (const auto &count : block)
      caliFile.CountMap.push_back(count);

  // 使用 binary_cast 序列化为二进制流
  auto header = CaliFileHeader();
  header.header.version = 1; // 设置版本号
  auto stream = openpni::autogen::add_header_and_binary_cast(header, caliFile);
  // auto binaryStream = openpni::autogen::binary::binary_cast(caliFile);

  // 写入文件
  std::ofstream file(filename, std::ios::binary);
  if (!file)
    throw openpni::exceptions::file_cannot_access();

  file << stream.rdbuf();
  if (!file.good())
    throw std::runtime_error("Failed to write calibration table to file: " + filename);
  file.close();
}

TempFrameV2 getTempFrameV2FromDataFrameV2(
    const DataFrameV2 &src) {
  TempFrameV2 result;
  result.energy = float((src.Energy[0] << 8) | src.Energy[1]);
  result.x = src.X;
  result.y = src.Y;
  result.du = src.nHeadAndDU & (0x03);
  return result;
}

void resolveOnePackage_simple(
    DataFrameV2 *begin, std::array<std::vector<TempFrameV2>, BLOCK_NUM> &output) {
  for (int i = 0; i < SINGLE_NUM_PER_PACKET; i++) {
    const auto tempFrame = getTempFrameV2FromDataFrameV2(begin[i]);
    if (tempFrame.du >= BLOCK_NUM) {
      // 如果 du 超出范围，跳过这个数据包
      continue;
    }
    if (tempFrame.x == 0 && tempFrame.y == 0 && tempFrame.energy == 0.0) {
      // 如果数据全为0，跳过这个数据包
      continue;
    }
    output[tempFrame.du].push_back(tempFrame);
  }
}
void loadData4Calibration_impl(
    const process::RawDataView &rawdata, uint16_t channelIndex,
    std::array<std::vector<TempFrameV2>, BLOCK_NUM> &output) {
  for (uint64_t i = 0; i < rawdata.count; i++) {
    if (rawdata.channel[i] != channelIndex)
      continue;
    if (rawdata.length[i] != UDP_PACKET_SIZE)
      continue;
    const auto begin = (DataFrameV2 *)(rawdata.data + rawdata.offset[i]);
    resolveOnePackage_simple(begin, output);
  }
}
BDM2Calibrater_impl::BDM2Calibrater_impl() {
  m_countMap = std::make_unique<decltype(m_countMap)::element_type>();
  m_centerPosition = std::make_unique<decltype(m_centerPosition)::element_type>();
  m_detectorChangable = DetectorChangable();
  isCalibrationLoaded = false;
}
void BDM2Calibrater_impl::appendData4Calibration(
    process::RawDataView rawdata, uint16_t channelIndex) {
  // for (auto &duData : m_tempFrame)
  //     duData.resize(0);
  loadData4Calibration_impl(rawdata, channelIndex, m_tempFrame);

  if (!m_countMap) {
    m_countMap = std::make_unique<std::array<CountMap_t, BLOCK_NUM>>();
    std::fill(m_countMap->begin(), m_countMap->end(), CountMap_t({0}));
  }
  for (const auto &duData : m_tempFrame)
    for (const auto &tempFrame : duData)
      (*m_countMap)[tempFrame.du][tempFrame.x + tempFrame.y * 256]++;
}

bool BDM2Calibrater_impl::setCalibrationInitialValue(
    std::array<uint8_t, CRYSTAL_LINE> _x, std::array<uint8_t, CRYSTAL_LINE> _y, uint8_t duIndex) {
  if (duIndex >= BLOCK_NUM)
    return false;
  if (!m_centerPosition)
    m_centerPosition = std::make_unique<std::array<CenterPosition_t, BLOCK_NUM>>();

  auto &centerPosition = (*m_centerPosition)[duIndex];
  std::sort(_x.begin(), _x.end());
  std::sort(_y.begin(), _y.end());

  for (const auto yIndex : std::views::iota(0ull, CRYSTAL_LINE))
    for (const auto xIndex : std::views::iota(0ull, CRYSTAL_LINE)) {
      centerPosition[xIndex + yIndex * CRYSTAL_LINE] = basic::make_vec2<float>(_x[xIndex], _y[yIndex]);
    }

  return true;
}

bool BDM2Calibrater_impl::setCalibrationInitialValue(
    std::array<basic::Vec2<float>, CRYSTAL_NUM_ONE_BLOCK> _p, uint8_t duIndex) {
  if (duIndex >= BLOCK_NUM)
    return false;
  if (!m_centerPosition)
    m_centerPosition = std::make_unique<std::array<CenterPosition_t, BLOCK_NUM>>();

  auto &centerPosition = (*m_centerPosition)[duIndex];
  centerPosition = _p;
  return true;
}

bool BDM2Calibrater_impl::setCalibrationInitialValue(
    uint8_t duIndex) {
  if (duIndex >= BLOCK_NUM)
    return false;
  return setCalibrationInitialValue(getAutoInitialX((*m_countMap)[duIndex]), getAutoInitialY((*m_countMap)[duIndex]),
                                    duIndex);
}

bool BDM2Calibrater_impl::generateCaliTable() {
  // 检查必要的成员变量是否已初始化
  if (!m_countMap || !m_centerPosition)
    return false;
  if (!m_calibrationResult) {
    m_calibrationResult = std::make_unique<CalibrationResult>();
    m_calibrationResult->energyCoef = std::make_unique<std::array<EnergyCoefs_t, BLOCK_NUM>>();
    m_calibrationResult->positionTable = std::make_unique<std::array<PositionTable_t, BLOCK_NUM>>();
    m_calibrationResult->centerPosition = std::make_unique<std::array<CenterPosition_t, BLOCK_NUM>>();
    m_calibrationResult->energyProfile =
        std::make_unique<std::array<std::array<EnergyProfile_t, CRYSTAL_NUM_ONE_BLOCK>, BLOCK_NUM>>();
    m_calibrationResult->countMap = std::make_unique<std::array<CountMap_t, BLOCK_NUM>>();
    m_calibrationResult->energyProfileCut =
        std::make_unique<std::array<std::array<EnergyProfileCut_t, CRYSTAL_NUM_ONE_BLOCK>, BLOCK_NUM>>();
  }

  constexpr int CONVOLUTION_KERNEL_HALF_SIZE = 5;
  const std::array<double, 2 * CONVOLUTION_KERNEL_HALF_SIZE + 1> kernel{1., 2., 3., 4., 5., 6., 5., 4., 3., 2., 1.};

  const auto originCurMin = 0.;
  const auto originCurMax = 65536.;

  try {
    for (int duIndex = 0; duIndex < BLOCK_NUM; duIndex++) {
      m_calibrationResult->countMap->at(duIndex) = (*m_countMap)[duIndex];
      KMeans<256, CRYSTAL_LINE, 4, float, uint8_t>((*m_countMap)[duIndex], (*m_centerPosition)[duIndex],
                                                   (*m_calibrationResult->centerPosition)[duIndex],
                                                   (*m_calibrationResult->positionTable)[duIndex]);
      (*m_calibrationResult->energyProfileCut)[duIndex].fill(EnergyProfileCut_t{originCurMin, originCurMax});
      generateEnergyCountMap<CONVOLUTION_KERNEL_HALF_SIZE>(
          (*m_calibrationResult->energyProfile)[duIndex], kernel, m_tempFrame[duIndex],
          (*m_calibrationResult->positionTable)[duIndex], (*m_calibrationResult->energyProfileCut)[duIndex]);
      for (int i = 0; i < CRYSTAL_NUM_ONE_BLOCK; i++) {
        const auto maxIndex = std::max_element((*m_calibrationResult->energyProfile)[duIndex][i].begin(),
                                               (*m_calibrationResult->energyProfile)[duIndex][i].end()) -
                              (*m_calibrationResult->energyProfile)[duIndex][i].begin();
        const auto cutMin = (*m_calibrationResult->energyProfileCut)[duIndex][i][0];
        const auto cutMax = (*m_calibrationResult->energyProfileCut)[duIndex][i][1];
        const auto maxValue =
            cutMin + (cutMax - cutMin) * maxIndex / (*m_calibrationResult->energyProfile)[duIndex][i].size();
        const auto newCutMin = 0.1 * maxValue;
        const auto newCutMax = 2.0 * maxValue;
        (*m_calibrationResult->energyProfileCut)[duIndex][i][0] = newCutMin;
        (*m_calibrationResult->energyProfileCut)[duIndex][i][1] = newCutMax;
      }
      generateEnergyCountMap<CONVOLUTION_KERNEL_HALF_SIZE>(
          (*m_calibrationResult->energyProfile)[duIndex], kernel, m_tempFrame[duIndex],
          (*m_calibrationResult->positionTable)[duIndex], (*m_calibrationResult->energyProfileCut)[duIndex]);

      for (int i = 0; i < CRYSTAL_NUM_ONE_BLOCK; i++) {
        const auto &countMap = (*m_calibrationResult->energyProfile)[duIndex][i];
        int maxValueIndex = 0;
        float maxValue = countMap[0];
        for (int j = 1; j < countMap.size(); j++) {
          if (countMap[j] > maxValue) {
            maxValue = countMap[j];
            maxValueIndex = j;
          }
        }
        const auto cutMin = (*m_calibrationResult->energyProfileCut)[duIndex][i][0];
        const auto cutMax = (*m_calibrationResult->energyProfileCut)[duIndex][i][1];
        (*m_calibrationResult->energyCoef)[duIndex][i] =
            511e3 / (cutMin + (cutMax - cutMin) * maxValueIndex / countMap.size());
      }
    }
  }

  catch (const std::exception &e) {
    // 如果发生异常，清理并返回 false
    m_calibrationResult.reset();
    return false;
  }

  return true;
}

void BDM2Calibrater_impl::adjustEnergyCoef(
    uint8_t duIndex, uint8_t crystalIndex, float peakEnergy) {
  if (!m_calibrationResult)
    throw std::runtime_error("CalibrationResult is not initialized.");
  if (duIndex >= BLOCK_NUM)
    throw std::out_of_range("duIndex out of range.");
  if (crystalIndex >= CRYSTAL_NUM_ONE_BLOCK)
    throw std::out_of_range("crystalIndex out of range.");
  // std::cout << "Before adjust: Core "
  //           << (*m_calibrationResult->energyCoef)[duIndex][crystalIndex] << " & ";
  (*m_calibrationResult->energyCoef)[duIndex][crystalIndex] *= (511e3 / peakEnergy);
  // std::cout << "After adjust: Core "
  //           << (*m_calibrationResult->energyCoef)[duIndex][crystalIndex] << std::endl;
}

const bdm2::CalibrationResult &BDM2Calibrater_impl::getCalibrationResult() const {
  if (!m_calibrationResult)
    throw std::runtime_error("CalibrationResult is not initialized.");
  return *m_calibrationResult;
}

CountMap_t BDM2Calibrater_impl::countMap(
    uint8_t duIndex) {
  if (duIndex >= BLOCK_NUM)
    return CountMap_t();
  if (!m_countMap)
    return CountMap_t();
  return (*m_countMap)[duIndex];
}

DetectorChangable &BDM2Calibrater_impl::detectorChangable() noexcept {
  return m_detectorChangable;
}

const DetectorChangable &BDM2Calibrater_impl::detectorChangable() const noexcept {
  return m_detectorChangable;
}
} // namespace openpni::device::bdm2

namespace openpni::device::bdm2 {
BDM2Runtime_impl::BDM2Runtime_impl() {
  m_h_caliTable = std::make_unique<CalibrationTable>();
  m_h_caliTable->energyCoef = std::make_unique<std::array<EnergyCoefs_t, BLOCK_NUM>>();
  m_h_caliTable->positionTable = std::make_unique<std::array<PositionTable_t, BLOCK_NUM>>();
  m_detectorChangable = DetectorChangable();
}

void BDM2Runtime_impl::loadCaliTable(
    const std::string &filename) {
  // 从文件中读取二进制数据
  auto stream = openpni::autogen::readBinaryFile(filename);
  if (!stream)
    throw openpni::exceptions::file_cannot_access();

  CaliFileHeader header;
  stream->seekg(0, std::ios_base::beg);
  stream->read(reinterpret_cast<char *>(&header), sizeof(header));
  if (!stream->good())
    throw openpni::exceptions::file_format_incorrect();
  // Skip sizeof(CaliFileHeader) bytes from the beginning of the stream
  auto stream_skip_header = openpni::autogen::subStream(*stream, sizeof(CaliFileHeader), stream->str().size());

#define try_assert(condition)                                                                                          \
  if (!(condition)) {                                                                                                  \
    throw openpni::exceptions::file_format_incorrect();                                                                \
  }

  try {
    if (header.header.version == 1) {
      auto caliFile =
          openpni::autogen::binary::struct_cast<openpni::autogen::binary::BDM2CaliFileV1>(stream_skip_header);
      m_h_caliTable = std::make_unique<CalibrationTable>();
      m_h_caliTable->energyCoef = std::make_unique<std::array<EnergyCoefs_t, BLOCK_NUM>>();
      m_h_caliTable->positionTable = std::make_unique<std::array<PositionTable_t, BLOCK_NUM>>();

      try_assert(caliFile.EnergyCoefs.size() == BLOCK_NUM * CRYSTAL_NUM_ONE_BLOCK);
      for (int i = 0; i < BLOCK_NUM; ++i) {
        std::copy(caliFile.EnergyCoefs.begin() + i * CRYSTAL_NUM_ONE_BLOCK,
                  caliFile.EnergyCoefs.begin() + (i + 1) * CRYSTAL_NUM_ONE_BLOCK,
                  (*m_h_caliTable->energyCoef)[i].begin());
      }

      try_assert(caliFile.PositionTable.size() == BLOCK_NUM * 256 * 256);
      for (int i = 0; i < BLOCK_NUM; ++i) {
        std::copy(caliFile.PositionTable.begin() + i * 256 * 256, caliFile.PositionTable.begin() + (i + 1) * 256 * 256,
                  (*m_h_caliTable->positionTable)[i].begin());
      }

#if !PNI_STANDARD_CONFIG_DISABLE_CUDA
      // 将数据加载到 CUDA 内存
      m_d_energyCoef = make_cuda_sync_ptr<float>(BLOCK_NUM * CRYSTAL_NUM_ONE_BLOCK);
      for (int i = 0; i < BLOCK_NUM; ++i) {
        cudaMemcpy(m_d_energyCoef.data() + i * CRYSTAL_NUM_ONE_BLOCK, (*m_h_caliTable->energyCoef)[i].data(),
                   CRYSTAL_NUM_ONE_BLOCK * sizeof(float), cudaMemcpyHostToDevice);
      }

      m_d_positionTable = make_cuda_sync_ptr<uint8_t>(BLOCK_NUM * 256 * 256);
      for (int i = 0; i < BLOCK_NUM; ++i) {
        cudaMemcpy(m_d_positionTable.data() + i * 256 * 256, (*m_h_caliTable->positionTable)[i].data(),
                   256 * 256 * sizeof(uint8_t), cudaMemcpyHostToDevice);
      }
#endif

      isCalibrationLoaded = true;
      return;
    }
  }

  catch (const std::exception &e) {
    throw openpni::exceptions::file_format_incorrect();
  }
  throw openpni::exceptions::file_unknown_version();
}

DetectorChangable &BDM2Runtime_impl::detectorChangable() noexcept {
  return m_detectorChangable;
}

const DetectorChangable &BDM2Runtime_impl::detectorChangable() const noexcept {
  return m_detectorChangable;
}

#if !PNI_STANDARD_CONFIG_DISABLE_CUDA
void BDM2Runtime_impl::r2s_cuda(
    const void *d_raw, const PacketPositionInfo *d_position, uint64_t count, basic::LocalSingle_t *d_out) const {
  r2s_cuda_impl(d_raw, d_position, count, d_out, m_d_energyCoef.get(), m_d_positionTable.get());
}
#endif

} // namespace openpni::device::bdm2

namespace openpni::device {
template <>
DetectorUnchangable detectorUnchangable<bdm2::BDM2Runtime>() {
  DetectorUnchangable result;
  result.maxUDPPacketSize = bdm2::MAX_UDP_PACKET_SIZE;
  result.minUDPPacketSize = bdm2::MIN_UDP_PACKET_SIZE;
  result.maxSingleNumPerPacket = bdm2::MAX_SINGLE_NUM_PER_PACKET;
  result.minSingleNumPerPacket = bdm2::MIN_SINGLE_NUM_PER_PACKET;
  result.geometry.blockNumU = bdm2::BLOCK_NUM;
  result.geometry.blockNumV = 1;
  result.geometry.blockSizeU = bdm2::BLOCK_PITCH;
  result.geometry.blockSizeV = bdm2::BLOCK_PITCH;
  result.geometry.crystalNumU = bdm2::CRYSTAL_LINE;
  result.geometry.crystalNumV = bdm2::CRYSTAL_LINE;
  result.geometry.crystalSizeU = bdm2::CRYSTAL_PITCH;
  result.geometry.crystalSizeV = bdm2::CRYSTAL_PITCH;
  return result;
}
} // namespace openpni::device
