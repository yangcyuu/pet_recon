#pragma once
#ifndef PNI_STANDARD_BASIC_AQUISITION_RAWFILE_INPUT
#define PNI_STANDARD_BASIC_AQUISITION_RAWFILE_INPUT
#include <fstream>
#include <future>
#include <list>
#include <stdint.h>

#include "include/io/RawDataIO.hpp"
#include "include/process/Acquisition.hpp"
namespace openpni::io::rawdata {
class RawFileInputImpl {
public:
  RawFileInputImpl() noexcept;
  ~RawFileInputImpl() noexcept;

public:
  void open(std::string path);
  uint32_t segmentNum() noexcept;
  RawdataSegment getSegment(uint32_t segmentIndex, uint32_t prefetchIndex) noexcept;
  std::string typeNameOfChannel(
      uint16_t channelIndex) const noexcept {
    if (channelIndex >= m_typeNameOfChannel.size())
      return "Unknown";
    return m_typeNameOfChannel[channelIndex];
  }
  SegmentHeader getSegmentHeader(uint32_t segmentIndex) noexcept;
  RawdataHeader getHeader() noexcept;

private:
  using PrefetchType = std::pair<uint32_t, RawdataSegment>;

private:
  RawdataSegment getOneSegment_impl(uint32_t segmentIndex);

private:
  struct SegmentInfo {
    SegmentHeader header;
    size_t offsetOfChannel;
    size_t offsetOfLength;
    size_t offsetOfOffset;
    size_t offsetOfData;
  };

private:
  std::ifstream m_file;
  size_t m_fileSize;
  RawdataHeader m_header;
  std::vector<SegmentInfo> m_segmentInfo;
  std::future<PrefetchType> m_prefetchFuture;

  std::vector<std::string> m_typeNameOfChannel; // 每个通道的类型名称
};
} // namespace openpni::io::rawdata
#endif // !PNI_STANDARD_BASIC_AQUISITION_RAWFILE