#include "RawFileInputImpl.hpp"

#include <cstring>
#include <iostream>

#include "include/detector/Detectors.hpp"
#include "include/io/IO.hpp"
#include "include/misc/HeaderWithSizeReserved.hpp"
#include "src/common/TemplateFunctions.hpp"
namespace openpni::io::rawdata {
using RawDataView = openpni::process::RawDataView;
RawFileInputImpl::RawFileInputImpl() noexcept {}

RawFileInputImpl::~RawFileInputImpl() noexcept {}

void readChannelTypeNames(
    std::ifstream &ifs, std::vector<std::string> &typeNames, uint16_t channelNum) {
  std::array<char, device::names::TYPE_MAX_LENGTH> typeNameBuffer;
  typeNames.resize(channelNum);
  for (uint16_t i = 0; i < channelNum; ++i) {
    ifs.read(typeNameBuffer.data(), typeNameBuffer.size());
    typeNames[i] = std::string(typeNameBuffer.data(), typeNameBuffer.end());
    // 去除末尾的空字符
    typeNames[i].erase(std::find(typeNames[i].begin(), typeNames[i].end(), '\0'), typeNames[i].end());
  }
}

void RawFileInputImpl::open(
    std::string path) {
  m_file.open(path, std::ios_base::binary);
  if (!m_file.is_open())
    throw openpni::exceptions::file_cannot_access();

  m_fileSize = m_file.seekg(0, std::ios_base::end).tellg();
  m_file.seekg(0, std::ios_base::beg);

  misc::HeaderWithSizeReserved<RawdataHeader, RAWDATA_FILE_HEADER_SIZE> H;

  m_file.read((char *)&H, sizeof(H));
  if (!m_file.good())
    throw openpni::exceptions::file_format_incorrect();
  m_header = H.header;
  if (::strcmp(m_header.fileTypeName, "PNI-RAWDATA"))
    throw openpni::exceptions::file_type_mismatch();

  readChannelTypeNames(m_file, m_typeNameOfChannel, m_header.channelNum);
  if (!m_file.good())
    throw openpni::exceptions::file_format_incorrect();

  size_t currentReadPosition = m_file.tellg();
  while (true) {
    SegmentInfo segment;
    misc::HeaderWithSizeReserved<SegmentHeader, RAWDATA_SEGMENT_HEADER_SIZE> s;
    auto &header = s.header;
    m_file.read((char *)&(s), sizeof(s));
    if (!m_file.good())
      throw openpni::exceptions::file_format_incorrect();
    segment.header = header;
    currentReadPosition = m_file.tellg();
    segment.offsetOfData = currentReadPosition;
    segment.offsetOfChannel = segment.offsetOfData + header.dataByteSize;
    segment.offsetOfLength =
        segment.offsetOfChannel + header.count * sizeof(std::remove_pointer<decltype(RawDataView::channel)>::type);
    segment.offsetOfOffset =
        segment.offsetOfLength + header.count * sizeof(std::remove_pointer<decltype(RawDataView::length)>::type);
    m_segmentInfo.push_back(segment);
    std::size_t nextReadPosiion =
        segment.offsetOfOffset + header.count * sizeof(std::remove_pointer<decltype(RawDataView::offset)>::type);
    if (nextReadPosiion > m_fileSize)
      throw openpni::exceptions::file_format_incorrect();
    if (nextReadPosiion == m_fileSize)
      break;
    m_file.seekg(nextReadPosiion, std::ios_base::beg);
  }
}

uint32_t RawFileInputImpl::segmentNum() noexcept {
  return m_segmentInfo.size();
}

RawdataSegment RawFileInputImpl::getSegment(
    uint32_t segmentIndex, uint32_t prefetchIndex) noexcept {
  if (m_prefetchFuture.valid()) {
    m_prefetchFuture.wait();
    auto [lastPrefetchIndex, lastPrefetchData] = m_prefetchFuture.get();
    if (prefetchIndex < m_segmentInfo.size())
      m_prefetchFuture = std::async(std::launch::async, [prefetchIndex, this] {
        return std::pair{prefetchIndex, this->getOneSegment_impl(prefetchIndex)};
      });
    if (lastPrefetchIndex == segmentIndex) {
      return std::move(lastPrefetchData);
    } else {
      return getOneSegment_impl(segmentIndex);
    }
  } else {
    auto result = getOneSegment_impl(segmentIndex);
    if (prefetchIndex < m_segmentInfo.size())
      m_prefetchFuture = std::async(std::launch::async, [prefetchIndex, this] {
        return std::pair{prefetchIndex, this->getOneSegment_impl(prefetchIndex)};
      });
    return result;
  }
}

SegmentHeader RawFileInputImpl::getSegmentHeader(
    uint32_t segmentIndex) noexcept {
  return m_segmentInfo[segmentIndex].header;
}

RawdataHeader RawFileInputImpl::getHeader() noexcept {
  return m_header;
}

RawdataSegment RawFileInputImpl::getOneSegment_impl(
    uint32_t segmentIndex) {
  const auto &segmentHeader = m_segmentInfo[segmentIndex].header;
  RawdataSegment result;

  using openpni::common::make_unique_uninitialized;

  make_unique_uninitialized(result.data, segmentHeader.dataByteSize);
  m_file.seekg(m_segmentInfo[segmentIndex].offsetOfData, std::ios_base::beg)
      .read((char *)result.data.get(), sizeof(decltype(result.data)::element_type) * segmentHeader.dataByteSize);
  make_unique_uninitialized(result.channel, segmentHeader.count);
  m_file.seekg(m_segmentInfo[segmentIndex].offsetOfChannel, std::ios_base::beg)
      .read((char *)result.channel.get(), sizeof(decltype(result.channel)::element_type) * segmentHeader.count);
  make_unique_uninitialized(result.length, segmentHeader.count);
  m_file.seekg(m_segmentInfo[segmentIndex].offsetOfLength, std::ios_base::beg)
      .read((char *)result.length.get(), sizeof(decltype(result.length)::element_type) * segmentHeader.count);
  make_unique_uninitialized(result.offset, segmentHeader.count);
  m_file.seekg(m_segmentInfo[segmentIndex].offsetOfOffset, std::ios_base::beg)
      .read((char *)result.offset.get(), sizeof(decltype(result.offset)::element_type) * segmentHeader.count);

  return result;
}

} // namespace openpni::io::rawdata
