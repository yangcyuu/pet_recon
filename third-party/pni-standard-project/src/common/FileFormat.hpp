#pragma once
#include <array>
#include <fstream>
#include <sstream>

#include "../autogen/autogen_xml.hpp"
namespace openpni::autogen {
constexpr int GeneralFileHeaderSize = 512;
struct GeneralFileHeader {
  unsigned version;
};
inline std::optional<std::stringstream> readBinaryFile(
    const std::string &filename) {
  std::ifstream file(filename, std::ios::binary);
  if (!file)
    return std::nullopt;
  std::stringstream buffer;
  buffer << file.rdbuf();
  return buffer;
}

template <typename Header, binary::IsBinarySerializable T>
inline std::stringstream add_header_and_binary_cast(
    const Header &__header, const T &__struct) noexcept {
  std::stringstream result;
  result.write(reinterpret_cast<const char *>(&__header), sizeof(__header));
  result << binary_cast(__struct).str(); // stream不能直接加入另一个stream，需要转换为字符串
  return result;
}
inline std::stringstream subStream(
    std::stringstream &__stream, std::size_t __begin, std::size_t __end) {
  std::stringstream result;
  if (__end <= __begin)
    return result; // Invalid range, return empty stream
  std::size_t streamSize = __stream.str().size();
  __stream.seekg(__begin, std::ios_base::beg);
  std::size_t length = __end - __begin;
  length = std::min(length,
                    streamSize - __begin); // Ensure we don't read beyond the stream size
  if (length == 0)
    return result; // No data to read, return empty stream
  result << __stream.str().substr(__begin, length);
  return result;
}

} // namespace openpni::autogen
