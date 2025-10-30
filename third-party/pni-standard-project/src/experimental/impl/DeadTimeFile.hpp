#pragma once
#include <format>
#include <vector>

#include "include/Exceptions.hpp"
#include "include/misc/HeaderWithSizeReserved.hpp"
#include "src/autogen/autogen_xml.hpp"
#include "src/common/FileFormat.hpp"
namespace openpni::experimental::node::impl {
class MichDeadTimeFile {
public:
  enum OpenMode { Read, Write };

public:
  MichDeadTimeFile(
      OpenMode mode)
      : m_mode(mode) {}
  ~MichDeadTimeFile() = default;

public:
  auto getCFDTTable() const { return m_cfdtTable; }
  auto getRTTable() const { return m_rtTable; }
  void setCFDTTable(
      std::vector<float> const &table) {
    m_cfdtTable = table;
  }
  void setRTTable(
      std::vector<float> const &table) {
    m_rtTable = table;
  }
  void open(std::string path);

private:
  void openRead(std::string path);
  void openWrite(std::string path);
  void readV1(std::stringstream &&stream);

private:
  std::vector<float> m_cfdtTable;
  std::vector<float> m_rtTable;

  OpenMode m_mode;
};

inline void MichDeadTimeFile::open(
    std::string path) {
  if (m_mode == OpenMode::Read) {
    openRead(path);
  } else if (m_mode == OpenMode::Write) {
    openWrite(path);
  } else {
    throw exceptions::algorithm_unexpected_condition("Unknown open mode.");
  }
}

inline void MichDeadTimeFile::openRead(
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

inline void MichDeadTimeFile::readV1(
    std::stringstream &&stream) {
  auto caliFile = openpni::autogen::binary::struct_cast<openpni::autogen::binary::DeadTimeCalibration>(stream);
  m_cfdtTable = std::move(caliFile.CFDT);
  m_rtTable = std::move(caliFile.ActualRt);
}

inline void MichDeadTimeFile::openWrite(
    std::string path) {
  std::ofstream file(path, std::ios::binary);
  if (!file)
    throw openpni::exceptions::file_cannot_access();

  using FileHeader =
      misc::HeaderWithSizeReserved<openpni::autogen::GeneralFileHeader, openpni::autogen::GeneralFileHeaderSize>;
  FileHeader header;
  header.header.version = 1;
  file.write(reinterpret_cast<const char *>(&header), sizeof(header));

  openpni::autogen::binary::DeadTimeCalibration caliFile;
  caliFile.CFDT = m_cfdtTable;
  caliFile.ActualRt = m_rtTable;
  auto stream = openpni::autogen::binary::binary_cast(caliFile);
  file << stream.rdbuf();
}

} // namespace openpni::experimental::node::impl