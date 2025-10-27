#include "include/experimental/file/NormFile.hpp"

#include <format>

#include "include/Exceptions.hpp"
#include "include/misc/HeaderWithSizeReserved.hpp"
#include "src/autogen/autogen_xml.hpp"
#include "src/common/FileFormat.hpp"

namespace openpni::experimental::file {
void MichNormalizationFile::open(
    std::string path) {
  if (m_mode == OpenMode::Read) {
    openRead(path);
  } else if (m_mode == OpenMode::Write) {
    openWrite(path);
  } else {
    throw exceptions::algorithm_unexpected_condition("Unknown open mode.");
  }
}
void MichNormalizationFile::openRead(
    std::string path) {
  auto stream = openpni::autogen::readBinaryFile(path);
  if (!stream)
    throw openpni::exceptions::file_cannot_access();

  using FileHeader =
      misc::HeaderWithSizeReserved<openpni::autogen::GeneralFileHeader, openpni::autogen::GeneralFileHeaderSize>;
  FileHeader header;
  stream->seekg(0, std::ios_base::beg);
  stream->read(reinterpret_cast<char *>(&header), sizeof(header));
  if (!stream->good())
    throw openpni::exceptions::file_format_incorrect();
  // Skip sizeof(FileHeader) bytes from the beginning of the stream
  auto stream_skip_header = openpni::autogen::subStream(*stream, sizeof(FileHeader), stream->str().size());

  if (header.header.version == 1) {
    readV1(std::move(stream_skip_header));
  } else {
    throw openpni::exceptions::file_format_incorrect(
        std::format("Expected version 1, got version {}", header.header.version));
  }
}
void MichNormalizationFile::readV1(
    std::stringstream &&stream) {
  auto caliFile = openpni::autogen::binary::struct_cast<openpni::autogen::binary::NormalizationFactorsV1>(stream);
  m_cryCount = std::move(caliFile.CryCount);
  m_blockFctA = std::move(caliFile.BlockA);
  m_blockFctT = std::move(caliFile.BlockT);
  m_planeFct = std::move(caliFile.Plane);
  m_radialFct = std::move(caliFile.Radial);
  m_interferenceFct = std::move(caliFile.Interference);
  m_cryFct = std::move(caliFile.CryFct);
}

void MichNormalizationFile::openWrite(
    std::string path) {
  openpni::autogen::binary::NormalizationFactorsV1 caliFile;
  caliFile.CryCount = m_cryCount;
  caliFile.BlockA = m_blockFctA;
  caliFile.BlockT = m_blockFctT;
  caliFile.Plane = m_planeFct;
  caliFile.Radial = m_radialFct;
  caliFile.Interference = m_interferenceFct;
  caliFile.CryFct = m_cryFct;

  // 使用 binary_cast 序列化为二进制流
  using FileHeader =
      misc::HeaderWithSizeReserved<openpni::autogen::GeneralFileHeader, openpni::autogen::GeneralFileHeaderSize>;
  auto header = FileHeader();
  header.header.version = 1; // 设置版本号
  auto stream = openpni::autogen::add_header_and_binary_cast(header, caliFile);
  // auto binaryStream = openpni::autogen::binary::binary_cast(caliFile);

  // 写入文件
  std::ofstream file(path, std::ios::binary);
  if (!file)
    throw openpni::exceptions::file_cannot_access();

  file << stream.rdbuf();
  if (!file.good())
    throw std::runtime_error("Failed to write normalization factors to file: " + path);
}
} // namespace openpni::experimental::file