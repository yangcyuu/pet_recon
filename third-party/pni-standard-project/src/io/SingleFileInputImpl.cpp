#include "SingleFileInputImpl.hpp"

#include "SingleFile.hpp"
#include "include/misc/HeaderWithSizeReserved.hpp"
namespace openpni::io::single {
SingleFileInput_impl::SingleFileInput_impl() noexcept {}
SingleFileInput_impl::~SingleFileInput_impl() noexcept {}
void SingleFileInput_impl::open(std::string path) {
  m_filePath = path;
  m_file.open(m_filePath, std::ios_base::binary);
  if (!m_file.is_open())
    throw openpni::exceptions::file_cannot_access();

  m_fileSize = m_file.seekg(0, std::ios_base::end).tellg();
  m_file.seekg(0, std::ios_base::beg);

  misc::HeaderWithSizeReserved<SingleFileHeader, SINGLE_FILE_HEADER_SIZE> H;
  m_file.read((char *)&H, sizeof(H));
  if (!m_file.good())
    throw openpni::exceptions::file_format_incorrect();

  m_header = H.header;
  if (::strcmp(m_header.fileTypeName, "PNI-SINGLE"))
    throw openpni::exceptions::file_type_mismatch();
  // Verify header flags
  const auto flags = invalidFlags(m_header);
  if (flags != SingleFileFlagsInvalid::NONE)
    throw flags;

  size_t currentReadPosition = m_file.tellg();
  while (true) {
    SingleSegmentInfo segment;
    std::optional<SingleSegmentHeader> segmentHeader =
        misc::HeaderWithSizeReserved<SingleSegmentHeader,
                                     SINGLE_SEGMENT_HEADER_SIZE>::readFromStream(m_file);
    if (!segmentHeader.has_value())
      throw openpni::exceptions::file_format_incorrect();

    segment.header = segmentHeader.value();
    currentReadPosition = m_file.tellg();

    segment.offsetOfCrystalIndex = currentReadPosition;
    segment.offsetOfTimeValue =
        segment.offsetOfCrystalIndex + m_header.bytes4CrystalIndex * segment.header.count;
    segment.offsetOfEnergy =
        segment.offsetOfTimeValue + m_header.bytes4TimeValue * segment.header.count;
    const auto nextReadPosition = currentReadPosition =
        segment.offsetOfEnergy + m_header.bytes4Energy * segment.header.count;
    m_segmentInfos.push_back(segment);

    if (nextReadPosition > m_fileSize)
      throw openpni::exceptions::file_format_incorrect();
    if (nextReadPosition == m_fileSize)
      break;
    m_file.seekg(nextReadPosition, std::ios_base::beg);
  }
}
SingleFileHeader SingleFileInput_impl::header() const noexcept {
  return m_header;
}
uint32_t SingleFileInput_impl::segmentNum() const noexcept {
  return m_segmentInfos.size();
}
SingleSegmentHeader
SingleFileInput_impl::segmentHeader(uint32_t segmentIndex) const noexcept {
  return m_segmentInfos[segmentIndex].header;
}

SingleSegmentBytes SingleFileInput_impl::readSegment(uint32_t segmentIndex,
                                                     uint32_t prefetchIndex) noexcept {
  if (m_prefetchFuture.valid()) {
    m_prefetchFuture.wait();
    auto [lastPrefetchIndex, lastPrefetchData] = m_prefetchFuture.get();
    if (prefetchIndex < m_segmentInfos.size())
      m_prefetchFuture = std::async(std::launch::async, [prefetchIndex, this] {
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
      m_prefetchFuture = std::async(std::launch::async, [prefetchIndex, this] {
        return std::pair{prefetchIndex, this->readSegment_impl(prefetchIndex)};
      });
    return result;
  }
}
SingleSegmentBytes SingleFileInput_impl::readSegment_impl(uint32_t segmentIndex) {
  if (segmentIndex >= m_segmentInfos.size())
    return {};

  const auto &segmentInfo = m_segmentInfos[segmentIndex];
  SingleSegmentBytes result;

  const auto bytes4CrystalIndex = m_header.bytes4CrystalIndex * segmentInfo.header.count;
  const auto bytes4TimeValue = m_header.bytes4TimeValue * segmentInfo.header.count;
  const auto bytes4Energy = m_header.bytes4Energy * segmentInfo.header.count;

  using openpni::common::make_unique_uninitialized;
  // 分配内存的时候要多分配512字节，以防止后续处理发生溢出
  if (bytes4CrystalIndex > 0) {
    result.storagedBytesCrystalIndex =
        segmentInfo.header.count * m_header.bytes4CrystalIndex + 512;
    make_unique_uninitialized(result.crystalIndexBytes, result.storagedBytesCrystalIndex);
    m_file.seekg(segmentInfo.offsetOfCrystalIndex, std::ios_base::beg)
        .read(result.crystalIndexBytes.get(), bytes4CrystalIndex);
  }
  if (bytes4TimeValue > 0) {
    result.storagedBytesTimeValue =
        segmentInfo.header.count * m_header.bytes4TimeValue + 512;
    make_unique_uninitialized(result.timeValueBytes, result.storagedBytesTimeValue);
    m_file.seekg(segmentInfo.offsetOfTimeValue, std::ios_base::beg)
        .read(result.timeValueBytes.get(), bytes4TimeValue);
  }
  if (bytes4Energy > 0) {
    result.storagedBytesEnergy = segmentInfo.header.count * m_header.bytes4Energy + 512;
    make_unique_uninitialized(result.energyBytes, result.storagedBytesEnergy);
    m_file.seekg(segmentInfo.offsetOfEnergy, std::ios_base::beg)
        .read(result.energyBytes.get(), bytes4Energy);
  }

  return result;
}

} // namespace openpni::io::single
