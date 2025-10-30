#include "ListmodeFileOutputImpl.hpp"
#include "ListmodeFile.hpp"
#include "include/misc/HeaderWithSizeReserved.hpp"
#include "src/common/UtilFunctions.hpp"
#include <filesystem>
namespace openpni::io::listmode {
ListmodeFileOutput_impl::ListmodeFileOutput_impl() noexcept {}
ListmodeFileOutput_impl::~ListmodeFileOutput_impl() noexcept {
  m_inputBuffer.stop();
  if (m_compressionThread.joinable())
    m_compressionThread.join();
  if (m_internalOutputThread.joinable())
    m_internalOutputThread.join();
}
bool ListmodeFileOutput_impl::setBytes4CrystalIndex1(CrystalIndexType type) noexcept {
  m_header.bytes4CrystalIndex1 = static_cast<uint8_t>(type);
  return listmode::isValidCrystalIndexType(type);
}
bool ListmodeFileOutput_impl::setBytes4CrystalIndex2(CrystalIndexType type) noexcept {
  m_header.bytes4CrystalIndex2 = static_cast<uint8_t>(type);
  return listmode::isValidCrystalIndexType(type);
}
bool ListmodeFileOutput_impl::setBytes4TimeValue1_2(TimeValue1_2Type type) noexcept {
  m_header.bytes4TimeValue1_2 = static_cast<uint8_t>(type);
  return listmode::isValidTimeValue1_2Type(type);
}

void ListmodeFileOutput_impl::open(std::string path) {
  const auto flags = listmode::invalidFlags(m_header);
  if (flags != ListmodeFileFlagsInvalid::NONE)
    throw flags;

  m_filePath = path;
  if (path.empty())
    throw openpni::exceptions::file_path_empty();
  m_file.open(path, std::ios_base::binary);
  if (!m_file.is_open())
    throw openpni::exceptions::file_cannot_access();
  m_header.version = 0;
  m_file.seekp(0, std::ios_base::beg);
  misc::HeaderWithSizeReserved<ListmodeFileHeader,
                               LISTMODE_FILE_HEADER_SIZE>::writeToStream(m_file,
                                                                         m_header);

  m_internalOutputThread = std::thread([&] {
    auto doConsume = [&](const common::ReuseableStream &buffer) noexcept {
      if (buffer.pos() == 0)
        // skip
        return;

      this->m_file.seekp(0, std::ios_base::end);
      this->m_file.write(buffer.data(), buffer.pos());
      m_bufferedBytes -= buffer.pos();
    };
    while (this->m_outputBuffer.read(doConsume))
      ;

    m_file.seekp(0, std::ios_base::beg);
    misc::HeaderWithSizeReserved<ListmodeFileHeader,
                                 LISTMODE_FILE_HEADER_SIZE>::writeToStream(m_file,
                                                                           m_header);
  });

  m_compressionThread = std::thread([&] noexcept {
    auto doConsume = [&](const InputBufferItem &item) noexcept {
      if (item.count == 0)
        // skip
        return;

      m_outputBuffer.write([&](common::ReuseableStream &buffer) {
        buffer.reuse();
        auto ptrCrystalIndex1 = std::make_unique_for_overwrite<
            basic::Listmode_t::typeof_globalCrystalIndex[]>(item.count);
        auto ptrCrystalIndex2 = std::make_unique_for_overwrite<
            basic::Listmode_t::typeof_globalCrystalIndex[]>(item.count);
        auto ptrTimeValue1_2 =
            std::make_unique_for_overwrite<basic::Listmode_t::typeof_timeValue1_2[]>(
                item.count);
        // Distribute data in structs into the pointers
        for (uint64_t i = 0; i < item.count; i++) {
          ptrCrystalIndex1[i] = item.data[i].globalCrystalIndex1;
          ptrCrystalIndex2[i] = item.data[i].globalCrystalIndex2;
          ptrTimeValue1_2[i] = item.data[i].time1_2pico;
        }
        // 数据写入顺序：1.SegmentHeader -> 2.[crystalIndex1] -> 3.[crystalIndex2]
        // -> 4.[timeValue1_2]

        misc::HeaderWithSizeReserved<ListmodeSegmentHeader, LISTMODE_SEGMENT_HEADER_SIZE>
            H(item.header);
        // Write Step 1: SegmentHeader
        buffer.write(reinterpret_cast<const char *>(&H), sizeof(H));
        // Write Step 2: crystalIndex1
        writeCrystalIndex2Buffer(ptrCrystalIndex1.get(), item.count, buffer,
                                 m_header.bytes4CrystalIndex1);
        // Write Step 3: crystalIndex2
        writeCrystalIndex2Buffer(ptrCrystalIndex2.get(), item.count, buffer,
                                 m_header.bytes4CrystalIndex2);
        // Write Step 4: timeValue1_2
        writeTimeValue1_2ToBuffer(ptrTimeValue1_2.get(), item.count, buffer);
      });
    };
    while (this->m_inputBuffer.read(doConsume))
      ;
    m_outputBuffer.stop();
  });
}

bool ListmodeFileOutput_impl::appendSegment(const basic::Listmode_t *data, uint64_t count,
                                            uint64_t clock, uint32_t duration) noexcept {
  if (!m_file.is_open())
    return false;

  bool writeSuccess = true;
  m_inputBuffer.write([&](InputBufferItem &buffer) {
    const auto segmentBytes = calculateSegmentBytes(count);
    if (common::getSpaceInfo_noexcept(m_filePath).available <
        m_reservedBytes + m_bufferedBytes + segmentBytes) {
      writeSuccess = false;
      buffer.count = 0;
      return;
    }

    m_bufferedBytes += segmentBytes;
    if (buffer.data.size() < count)
      buffer.data.resize(count);
    buffer.count = count;
    buffer.header.count = count;
    buffer.header.clock = clock;
    buffer.header.duration = duration;
    ::memcpy(buffer.data.data(), data, count * sizeof(basic::Listmode_t));
  });

  if (writeSuccess)
    m_header.segmentNum++;
  return writeSuccess;
}

uint64_t ListmodeFileOutput_impl::calculateSegmentBytes(uint64_t count) const noexcept {
  uint64_t bytes = LISTMODE_SEGMENT_HEADER_SIZE;
  bytes += count * m_header.bytes4CrystalIndex1; // crystalIndex1
  bytes += count * m_header.bytes4CrystalIndex2; // crystalIndex2
  bytes += count * m_header.bytes4TimeValue1_2;  // timeValue1_2
  return bytes;
}

void ListmodeFileOutput_impl::writeCrystalIndex2Buffer(
    openpni::basic::Listmode_t::typeof_globalCrystalIndex *ptrCrystalIndex,
    uint64_t count, common::ReuseableStream &buffer,
    uint8_t bytes4EachElement) const noexcept {
  switch (CrystalIndexType(bytes4EachElement)) {
  case CrystalIndexType::UINT16: {
    auto cuttedCrystalIndex16 =
        common::cut_to_bytes<basic::Listmode_t::typeof_globalCrystalIndex, 2>(
            ptrCrystalIndex, ptrCrystalIndex + count);
    buffer.write(cuttedCrystalIndex16.get(), count * 2);
  } break;
  case CrystalIndexType::UINT24: {
    auto cuttedCrystalIndex24 =
        common::cut_to_bytes<basic::Listmode_t::typeof_globalCrystalIndex, 3>(
            ptrCrystalIndex, ptrCrystalIndex + count);
    buffer.write(cuttedCrystalIndex24.get(), count * 3);
  } break;
  case CrystalIndexType::UINT32:
    buffer.write(reinterpret_cast<const char *>(ptrCrystalIndex), count * 4);
    break;
  default:
    // Do nothing, as the flag is invalid
    break;
  }
}

void ListmodeFileOutput_impl::writeTimeValue1_2ToBuffer(
    openpni::basic::Listmode_t::typeof_timeValue1_2 *ptrTimeValue, uint64_t count,
    common::ReuseableStream &buffer) const noexcept {
  switch (TimeValue1_2Type(m_header.bytes4TimeValue1_2)) {
  case TimeValue1_2Type::ZERO:
    // Do nothing, as the there is no bytes to write
    break;
  case TimeValue1_2Type::INT8: {
    auto timeValueInt8 = std::make_unique_for_overwrite<char[]>(count);
    for (uint64_t i = 0; i < count; i++)
      timeValueInt8[i] = static_cast<char>(ptrTimeValue[i] >> 8);
    buffer.write(timeValueInt8.get(), count);
  } break;
  case TimeValue1_2Type::INT16:
    buffer.write(reinterpret_cast<const char *>(ptrTimeValue), count * 2);
    break;
  default:
    // Do nothing, as the flag is invalid
    break;
  }
}

} // namespace openpni::io::listmode
