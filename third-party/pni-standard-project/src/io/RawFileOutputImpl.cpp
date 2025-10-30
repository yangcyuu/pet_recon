#include "RawFileOutputImpl.hpp"

#include <cstring>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <type_traits>
#include <vector>

#include "include/Exceptions.hpp"
#include "include/detector/Detectors.hpp"
#include "include/io/IO.hpp"
#include "include/misc/HeaderWithSizeReserved.hpp"
#include "src/common/UtilFunctions.hpp"
namespace openpni::io::rawdata {
RawFileOutputImpl::RawFileOutputImpl() {}

RawFileOutputImpl::~RawFileOutputImpl() {
  m_inputBuffer.stop();
  if (m_internalCompressionThread.joinable())
    m_internalCompressionThread.join();
  if (m_internalOutputThread.joinable())
    m_internalOutputThread.join();
}

void writeChannelTypeNames(std::ofstream &ofs,
                           const std::vector<std::string> &typeNames) {
  std::array<char, device::names::TYPE_MAX_LENGTH> typeNameBuffer;
  for (const auto typeName : typeNames) {
    typeNameBuffer.fill(0);
    ::memcpy(typeNameBuffer.data(), typeName.c_str(),
             std::min(typeName.size(), typeNameBuffer.size() - 1));
    ofs.write(typeNameBuffer.data(), typeNameBuffer.size());
  }
}

void RawFileOutputImpl::open(std::string path) {
  if (path.empty())
    throw openpni::exceptions::file_path_empty();
  m_filePath = path;
  m_file.open(path, std::ios_base::binary);
  if (!m_file.is_open())
    throw openpni::exceptions::file_cannot_access();

  m_header.version = 0;
  m_file.seekp(0, std::ios_base::beg);
  misc::HeaderWithSizeReserved<RawdataHeader, RAWDATA_FILE_HEADER_SIZE>::writeToStream(
      m_file, m_header);
  writeChannelTypeNames(m_file, m_typeNameOfChannel);

  m_internalOutputThread = std::thread([&] noexcept {
    auto doConsume = [&](const OutputBufferItem &buffer) noexcept {
      if (buffer.buffer.pos() == 0)
        // skip
        return;

      this->m_file.seekp(0, std::ios_base::end);
      this->m_file.write(buffer.buffer.data(), buffer.buffer.pos());
      this->m_bufferedBytes -= buffer.bufferedBytes;
    };
    while (this->m_outputBuffer.read(doConsume))
      ;
    m_file.seekp(0, std::ios_base::beg);
    misc::HeaderWithSizeReserved<RawdataHeader, RAWDATA_FILE_HEADER_SIZE>::writeToStream(
        m_file, m_header);
  });

  m_internalCompressionThread = std::thread([&] noexcept {
    auto doConsume = [&](const InputBufferItem &item) noexcept {
      if (item.header.count == 0)
        return;

      this->m_outputBuffer.write([&](OutputBufferItem &buffer) {
        buffer.buffer.reuse();
        if (item.header.count == 0)
          return; // No valid data to write
        decltype(item.header) headerCopy = item.header;
        headerCopy.count = 0;
        headerCopy.dataByteSize = 0;
        misc::HeaderWithSizeReserved<SegmentHeader, RAWDATA_SEGMENT_HEADER_SIZE> H(
            headerCopy);
        buffer.buffer.write(reinterpret_cast<const char *>(&H), sizeof(H));
        const auto channelNum = m_header.channelNum;
        const auto count = item.header.count;

        uint64_t currentIndex = 0;
        uint64_t totalDataBytes = 0;
        std::vector<uint64_t> newOffsetVector(count);
        std::vector<uint16_t> newChannelVector(count);
        std::vector<uint16_t> newLengthVector(count);
        for (int i = 0; i < count; i++) {
          if (item.channel[i] >= channelNum)
            continue;

          buffer.buffer.write(reinterpret_cast<const char *>(&item.data[totalDataBytes]),
                              item.length[i]);
          newOffsetVector[currentIndex] = totalDataBytes;
          newChannelVector[currentIndex] = item.channel[i];
          newLengthVector[currentIndex] = item.length[i];
          totalDataBytes += item.length[i];
          currentIndex++;
        }
        const auto posAfterData = buffer.buffer.pos();

        if (currentIndex == 0)
          return; // No valid data to write
        buffer.buffer.setPos(0);
        H.header.count = currentIndex;
        H.header.dataByteSize = totalDataBytes;
        buffer.buffer.write(reinterpret_cast<const char *>(&H), sizeof(H));
        buffer.buffer.setPos(posAfterData);
        buffer.buffer.write(reinterpret_cast<const char *>(newChannelVector.data()),
                            currentIndex * sizeof(uint16_t));
        buffer.buffer.write(reinterpret_cast<const char *>(newLengthVector.data()),
                            currentIndex * sizeof(uint16_t));
        buffer.buffer.write(reinterpret_cast<const char *>(newOffsetVector.data()),
                            currentIndex * sizeof(uint64_t));

        buffer.bufferedBytes = item.bufferedBytes;
      });
    };
    while (this->m_inputBuffer.read(doConsume))
      ;
    m_outputBuffer.stop();
  });
}

bool RawFileOutputImpl::appendSegment(RawDataView rawData) noexcept {
  if (!m_file.is_open())
    return false;
  auto &header = m_header;
  bool writeSuccess = true;
  m_inputBuffer.write([&](InputBufferItem &item) {
    item.header.clock = rawData.clock_ms;
    item.header.duration = rawData.duration_ms;
    item.header.count = rawData.count;
    item.header.dataByteSize = 0;
    if (rawData.count == 0)
      return;

    const auto bytesUncompressedData = rawData.offset[rawData.count - 1] +
                                       rawData.length[rawData.count - 1] -
                                       rawData.offset[0];
    if (common::getSpaceInfo_noexcept(m_filePath).available <
        m_reservedStorage + m_bufferedBytes + bytesUncompressedData) {
      writeSuccess = false;
      item.header.count = 0; // No valid data to write
      return;
    }
    if (item.data.size() < bytesUncompressedData)
      item.data.resize(bytesUncompressedData);
    if (item.length.size() < rawData.count)
      item.length.resize(rawData.count);
    if (item.offset.size() < rawData.count)
      item.offset.resize(rawData.count);
    if (item.channel.size() < rawData.count)
      item.channel.resize(rawData.count);

    ::memcpy(item.data.data(), &rawData.data[rawData.offset[0]], bytesUncompressedData);
    ::memcpy(item.length.data(), rawData.length, rawData.count * sizeof(uint16_t));
    ::memcpy(item.offset.data(), rawData.offset, rawData.count * sizeof(uint64_t));
    ::memcpy(item.channel.data(), rawData.channel, rawData.count * sizeof(uint16_t));
    item.bufferedBytes =
        bytesUncompressedData + rawData.count * (sizeof(uint16_t) * 2 + sizeof(uint64_t));
    m_bufferedBytes += item.bufferedBytes;
  });

  if (writeSuccess)
    header.segmentNum++;
  return writeSuccess;
}
} // namespace openpni::io::rawdata
