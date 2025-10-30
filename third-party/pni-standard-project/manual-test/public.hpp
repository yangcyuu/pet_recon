#pragma once
#include <format>
#include <fstream>
#include <memory>

#include "include/experimental/core/Mich.hpp"
inline auto E180() {
  using namespace openpni::experimental;
  core::MichDefine mich;
  auto &polygon = mich.polygon;
  polygon.edges = 24;
  polygon.detectorPerEdge = 1;
  polygon.detectorLen = 0;
  polygon.radius = 106.5; // 这里在V3Norm中是114,与一般的106.5不同
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
inline auto E180_shell() {
  auto e180 = E180();
  e180.polygon.radius = 114.0f;
  return e180;
}

inline auto _930() {
  using namespace openpni::experimental;
  core::MichDefine mich;
  auto &polygon = mich.polygon;
  polygon.edges = 48;
  polygon.detectorPerEdge = 1;
  polygon.detectorLen = 0;
  polygon.radius = 807.6 / 2; // 这里在V3Norm中是114,与一般的106.5不同
  polygon.angleOf1stPerp = 0;
  polygon.detectorRings = 3;
  polygon.ringDistance = 25.5f * 4;
  auto &detector = mich.detector;
  detector.blockNumU = 4;
  detector.blockNumV = 2;
  detector.blockSizeU = 25.5f;
  detector.blockSizeV = 25.5f;
  detector.crystalNumU = 6;
  detector.crystalNumV = 6;
  detector.crystalSizeU = 4.2;
  detector.crystalSizeV = 4.2;
  return mich;
}

template <typename T>
inline std::unique_ptr<T[]> read_from_file(
    const std::string &file_path, size_t expectedElements, std::size_t offset = 0) {
  std::ifstream file(file_path, std::ios::binary);
  if (!file)
    throw std::runtime_error("Failed to open file: " + file_path);
  file.seekg(0, std::ios::end);
  size_t fileSize = file.tellg();
  if (fileSize != expectedElements * sizeof(T) + offset)
    throw std::runtime_error(
        std::format("File size mismatch: expected {} bytes, got {} bytes", expectedElements * sizeof(T), fileSize));
  file.seekg(offset, std::ios::beg);
  auto data = std::make_unique_for_overwrite<T[]>(expectedElements);
  file.read(reinterpret_cast<char *>(data.get()), expectedElements * sizeof(T));
  if (!file)
    throw std::runtime_error("Failed to read data from file: " + file_path);
  return data;
}

template <typename T>
void write_to_file(
    const std::string &file_path, const T *data, size_t elements) {
  std::ofstream outFile(file_path, std::ios::binary);
  if (!outFile)
    throw std::runtime_error("Failed to open file for writing: " + file_path);
  outFile.write(reinterpret_cast<const char *>(data), elements * sizeof(T));
  if (!outFile)
    throw std::runtime_error("Failed to write data to file: " + file_path);
}
template <typename T>
void write_to_file(
    const std::string &file_path, const std::unique_ptr<T[]> &data, size_t elements) {
  write_to_file(file_path, data.get(), elements);
}