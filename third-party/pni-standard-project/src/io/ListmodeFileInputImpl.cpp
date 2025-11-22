#include "ListmodeFileInputImpl.hpp"

#include "ListmodeFile.hpp"
#include "include/misc/HeaderWithSizeReserved.hpp"
namespace openpni::io::listmode {
ListmodeFileInput_impl::ListmodeFileInput_impl() noexcept {}
ListmodeFileInput_impl::~ListmodeFileInput_impl() noexcept {}
void ListmodeFileInput_impl::open(
    std::string path) {
  if (path.empty())
    throw openpni::exceptions::file_path_empty();
  m_filePath = path;
  m_file.open(path, std::ios_base::binary);
  if (!m_file.is_open())
    throw openpni::exceptions::file_cannot_access();
  m_fileSize = m_file.seekg(0, std::ios_base::end).tellg();
  m_file.seekg(0, std::ios_base::beg);

  std::optional<ListmodeFileHeader> header =
      misc::HeaderWithSizeReserved<ListmodeFileHeader, LISTMODE_FILE_HEADER_SIZE>::readFromStream(m_file);
  if (!header)
    throw openpni::exceptions::file_format_incorrect();
  if (::strcmp(m_header.fileTypeName, "PNI-LSTMODE"))
    throw openpni::exceptions::file_format_incorrect();

  m_header = header.value();
  const auto flags = listmode::invalidFlags(m_header);
  if (flags != ListmodeFileFlagsInvalid::NONE)
    throw flags;

  size_t currentReadPosition = m_file.tellg();
  while (true) {
    SegmentInfo segmentInfo;
    auto &segmentHeader = segmentInfo.header;
    if (!misc::HeaderWithSizeReserved<ListmodeSegmentHeader, LISTMODE_SEGMENT_HEADER_SIZE>::readFromStream(
            m_file, segmentHeader))
      throw openpni::exceptions::file_format_incorrect();

    currentReadPosition = m_file.tellg();
    segmentInfo.offsetOfCrystalIndex1 = currentReadPosition;
    segmentInfo.offsetOfCrystalIndex2 = currentReadPosition + segmentHeader.count * m_header.bytes4CrystalIndex1;
    segmentInfo.offsetOfTimeValue1_2 =
        segmentInfo.offsetOfCrystalIndex2 + segmentHeader.count * m_header.bytes4CrystalIndex2;

    size_t nextReadPosition = currentReadPosition =
        segmentInfo.offsetOfTimeValue1_2 + segmentHeader.count * m_header.bytes4TimeValue1_2;
    if (nextReadPosition > m_fileSize)
      throw openpni::exceptions::file_format_incorrect();
    m_segmentInfos.push_back(segmentInfo);
    if (nextReadPosition == m_fileSize)
      break;
    m_file.seekg(nextReadPosition, std::ios_base::beg);
  }
}

uint32_t ListmodeFileInput_impl::segmentNum() const noexcept {
  return static_cast<uint32_t>(m_segmentInfos.size());
}
ListmodeFileHeader ListmodeFileInput_impl::header() const noexcept {
  return m_header;
}
ListmodeSegmentHeader ListmodeFileInput_impl::segmentHeader(
    uint32_t segmentIndex) const noexcept {

  return m_segmentInfos[segmentIndex].header;
}
ListmodeSegmentBytes ListmodeFileInput_impl::readSegment(
    uint32_t segmentIndex, uint32_t prefetchIndex) noexcept {
  if (segmentIndex >= m_segmentInfos.size())
    return {};

  if (m_prefetchFuture.valid()) {
    m_prefetchFuture.wait();
    auto [lastPrefetchIndex, lastPrefetchData] = m_prefetchFuture.get();
    if (prefetchIndex < m_segmentInfos.size())
      m_prefetchFuture = std::async(std::launch::async, [prefetchIndex, this] noexcept {
        return std::pair{prefetchIndex, this->readSegment_impl(prefetchIndex)};
      });
    if (lastPrefetchIndex == segmentIndex) {
      return std::move(lastPrefetchData);
    } else {
      return readSegment_impl(segmentIndex);
    }
  } else {
    auto result = readSegment_impl(segmentIndex);
    if (prefetchIndex < m_segmentInfos.size())
      m_prefetchFuture = std::async(std::launch::async, [prefetchIndex, this] noexcept {
        return std::pair{prefetchIndex, this->readSegment_impl(prefetchIndex)};
      });
    return result;
  }
}

ListmodeSegmentBytes ListmodeFileInput_impl::readSegment_impl(
    uint32_t segmentIndex) {
  if (segmentIndex >= m_segmentInfos.size())
    return {};

  const auto &segmentInfo = m_segmentInfos[segmentIndex];
  ListmodeSegmentBytes result;
  result.storagedBytesCrystalIndex1 = segmentInfo.header.count * m_header.bytes4CrystalIndex1 + 512;
  result.storagedBytesCrystalIndex2 = segmentInfo.header.count * m_header.bytes4CrystalIndex2 + 512;
  result.storagedBytesTimeValue1_2 = segmentInfo.header.count * m_header.bytes4TimeValue1_2 + 512;

  using openpni::common::make_unique_uninitialized;
  make_unique_uninitialized(result.crystalIndex1Bytes, result.storagedBytesCrystalIndex1);
  make_unique_uninitialized(result.crystalIndex2Bytes, result.storagedBytesCrystalIndex2);
  make_unique_uninitialized(result.timeValue1_2Bytes, result.storagedBytesTimeValue1_2);

  const auto bytes4CrystalIndex1 = m_header.bytes4CrystalIndex1 * segmentInfo.header.count;
  const auto bytes4CrystalIndex2 = m_header.bytes4CrystalIndex2 * segmentInfo.header.count;
  const auto bytes4TimeValue1_2 = m_header.bytes4TimeValue1_2 * segmentInfo.header.count;

  if (bytes4CrystalIndex1 > 0) {
    m_file.seekg(segmentInfo.offsetOfCrystalIndex1, std::ios_base::beg)
        .read(result.crystalIndex1Bytes.get(), bytes4CrystalIndex1);
  }
  if (bytes4CrystalIndex2 > 0) {
    m_file.seekg(segmentInfo.offsetOfCrystalIndex2, std::ios_base::beg)
        .read(result.crystalIndex2Bytes.get(), bytes4CrystalIndex2);
  }
  if (bytes4TimeValue1_2 > 0) {
    m_file.seekg(segmentInfo.offsetOfTimeValue1_2, std::ios_base::beg)
        .read(result.timeValue1_2Bytes.get(), bytes4TimeValue1_2);
  }
  return result;
}
} // namespace openpni::io::listmode
