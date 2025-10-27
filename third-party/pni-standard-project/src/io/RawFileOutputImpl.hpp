#pragma once
#ifndef PNI_STANDARD_BASIC_AQUISITION_RAWFILE_OUTPUT
#define PNI_STANDARD_BASIC_AQUISITION_RAWFILE_OUTPUT
#include <fstream>
#include <list>
#include <stdint.h>
#include <thread>

#include "include/io/RawDataIO.hpp"
#include "include/misc/CycledBuffer.hpp"
#include "include/process/Acquisition.hpp"
#include "src/common/ReuseableStream.hpp"
namespace openpni::io::rawdata {
using RawDataView = openpni::process::RawDataView;

class RawFileOutputImpl {
public:
  RawFileOutputImpl();
  ~RawFileOutputImpl();

public:
  void open(std::string path);
  void setChannelNum(
      uint16_t channelNum) noexcept {
    m_header.channelNum = channelNum;
    m_typeNameOfChannel.resize(channelNum, "Unknown");
  }
  bool appendSegment(RawDataView rawData) noexcept;
  void setReservedBytes(
      uint64_t r) noexcept {
    m_reservedStorage = r;
  }
  void setTypeNameOfChannel(
      uint16_t channelIndex, const std::string &typeName) noexcept {
    m_typeNameOfChannel[channelIndex] = typeName;
  }

  // 配置字段
private:
  struct InputBufferItem {
    SegmentHeader header;
    std::vector<uint8_t> data;
    std::vector<uint16_t> length;
    std::vector<uint64_t> offset;
    std::vector<uint16_t> channel;
    uint64_t bufferedBytes{0};
  };
  struct OutputBufferItem {
    common::ReuseableStream buffer;
    uint64_t bufferedBytes{0};
  };

public:
  uint64_t m_reservedStorage{1024ull * 1024ull * 1024ull * 20ull}; // 默认保留空间为20GB

private:
  std::string m_filePath;
  std::ofstream m_file;
  RawdataHeader m_header;
  common::CycledBuffer<InputBufferItem> m_inputBuffer{2};
  common::CycledBuffer<OutputBufferItem> m_outputBuffer{2};
  std::thread m_internalCompressionThread;
  std::thread m_internalOutputThread;
  std::atomic<uint64_t> m_bufferedBytes{0};

  std::vector<std::string> m_typeNameOfChannel; // 每个通道的类型名称
};
} // namespace openpni::io::rawdata
#endif // !PNI_STANDARD_BASIC_AQUISITION_RAWFILE