#include "SingleFileOutputImpl.hpp"

#include <filesystem>

#include "SingleFile.hpp"
#include "include/Exceptions.hpp"
#include "include/io/IO.hpp"
#include "include/misc/HeaderWithSizeReserved.hpp"
#include "src/common/TemplateFunctions.hpp"
#include "src/common/UtilFunctions.hpp"
namespace openpni::io::single {
SingleFileOutput_impl::SingleFileOutput_impl() noexcept {}

SingleFileOutput_impl::~SingleFileOutput_impl() noexcept {
  m_inputBuffer.stop();
  if (m_internalCompressionThread.joinable())
    m_internalCompressionThread.join();
  if (m_internalOutputThread.joinable())
    m_internalOutputThread.join();
}
bool SingleFileOutput_impl::setBytes4CrystalIndex(CrystalIndexType type) noexcept {
  m_header.bytes4CrystalIndex = static_cast<uint8_t>(type);
  return isValidCrystalIndexType(type);
}
bool SingleFileOutput_impl::setBytes4TimeValue(TimeValueType type) noexcept {
  m_header.bytes4TimeValue = static_cast<uint8_t>(type);
  return isValidTimeValueType(type);
}
bool SingleFileOutput_impl::setBytes4Energy(EnergyType type) noexcept {
  m_header.bytes4Energy = static_cast<uint8_t>(type);
  return isValidEnergyType(type);
}
void SingleFileOutput_impl::setTotalCrystalNum(uint32_t num) noexcept {
  m_header.cystalNum = num;
}
void SingleFileOutput_impl::setReservedBytes(uint64_t r) noexcept {
  m_reversedBytes = r;
}

void SingleFileOutput_impl::open(std::string path) {
  const auto flags = invalidFlags(m_header);
  if (flags != SingleFileFlagsInvalid::NONE)
    throw flags;

  m_filePath = path;
  if (path.empty())
    throw openpni::exceptions::file_path_empty();

  m_file.open(path, std::ios_base::binary);
  if (!m_file.is_open())
    throw openpni::exceptions::file_cannot_access();
  m_header.version = 0;

  m_file.seekp(0, std::ios_base::beg);
  misc::HeaderWithSizeReserved<SingleFileHeader, SINGLE_FILE_HEADER_SIZE>::writeToStream(
      m_file, m_header);

  m_internalOutputThread = std::thread([&] noexcept {
    auto doCosume = [&](const common::ReuseableStream &buffer) {
      if (buffer.pos() == 0)
        // skip
        return;

      this->m_file.seekp(0, std::ios_base::end);
      this->m_file.write(buffer.data(), buffer.pos());
      m_bufferedBytes -= buffer.pos();
    };

    while (this->m_outputBuffer.read(doCosume))
      ; // Read from the output buffer and write to the file, until the buffer is
        // end-of-file

    m_file.seekp(0, std::ios_base::beg);
    misc::HeaderWithSizeReserved<SingleFileHeader,
                                 SINGLE_FILE_HEADER_SIZE>::writeToStream(m_file,
                                                                         m_header);
  });

  m_internalCompressionThread = std::thread([&] noexcept {
    auto doConsume = [&](const InputBufferItem &item) noexcept {
      if (item.count == 0)
        // skip
        return;

      m_outputBuffer.write([&](common::ReuseableStream &buffer) {
        buffer.reuse();
        auto ptrCrystalIndex =
            std::make_unique<basic::GlobalSingle_t::typeof_globalCrystalIndex[]>(
                item.count);
        auto ptrTimeValue =
            std::make_unique<basic::GlobalSingle_t::typeof_timeValue_pico[]>(item.count);
        auto ptrEnergy =
            std::make_unique<basic::GlobalSingle_t::typeof_energy[]>(item.count);
        // Distribute data in structs into the pointers
        for (uint64_t i = 0; i < item.count; i++) {
          ptrCrystalIndex[i] = item.data[i].globalCrystalIndex;
          ptrTimeValue[i] = item.data[i].timeValue_pico;
          ptrEnergy[i] = item.data[i].energy;
        }
        // 数据写入顺序：1.SegmentHeader -> 2.[crystalIndex] -> 3.[timeValue]
        // -> 4.[energy]
        misc::HeaderWithSizeReserved<SingleSegmentHeader, SINGLE_SEGMENT_HEADER_SIZE> H(
            item.header);
        // Write Step 1: SegmentHeader
        buffer.write(reinterpret_cast<const char *>(&H), sizeof(H));
        // Write Step 2: crystalIndex
        write2BufferCrystalIndex(ptrCrystalIndex, item.count, buffer);
        // Write Step 3: timeValue
        write2BufferTimeValue(ptrTimeValue, item.count, buffer);
        // Write Step 4: energy
        write2BufferEnergy(item.count, ptrEnergy, buffer);
      });
    };
    while (this->m_inputBuffer.read(doConsume))
      ; // Read from the input buffer and write to the output buffer, until the buffer is
        // end-of-file

    m_outputBuffer.stop();
  });
}

bool SingleFileOutput_impl::appendSegment(const basic::GlobalSingle_t *data,
                                          uint64_t count, uint64_t clock,
                                          uint32_t duration) noexcept {
  if (!m_file.is_open())
    return false;

  bool writeSuccess = true;
  m_inputBuffer.write([&](InputBufferItem &buffer) {
    const auto segmentBytes = calculateSegmentBytes(count);
    if (common::getSpaceInfo_noexcept(m_filePath).available <
        m_reversedBytes + m_bufferedBytes + calculateSegmentBytes(count)) {
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

    ::memcpy(buffer.data.data(), data, count * sizeof(basic::GlobalSingle_t));
  });

  if (writeSuccess)
    m_header.segmentNum++;

  return writeSuccess;
}

void single::SingleFileOutput_impl::write2BufferEnergy(
    uint64_t count,
    std::unique_ptr<openpni::basic::GlobalSingle::typeof_energy[]> &ptrEnergy,
    openpni::common::ReuseableStream &buffer) {
  switch (EnergyType(m_header.bytes4Energy)) {
  case EnergyType::ZERO:
    // Do nothing, as no energy data is stored
    break;
  case EnergyType::UINT8: {
    auto cuttedEnergy8 = std::make_unique<uint8_t[]>(count);
    for (uint64_t i = 0; i < count; i++)
      cuttedEnergy8[i] =
          ptrEnergy[i] / 4e3; // Range: up to 1024keV, so divide by 4e3 to fit in uint8_t
    buffer.write(reinterpret_cast<const char *>(cuttedEnergy8.get()),
                 count * sizeof(uint8_t));
  } break;
  case EnergyType::UFLT16: {
    auto cuttedEnergy16 = std::make_unique<basic::UFloat16_t[]>(count);
    for (uint64_t i = 0; i < count; i++)
      cuttedEnergy16[i] = basic::flt32_flt16(ptrEnergy[i]);
    buffer.write(reinterpret_cast<const char *>(cuttedEnergy16.get()),
                 count * sizeof(basic::UFloat16_t));
  } break;
  case EnergyType::FLT32:
    buffer.write(reinterpret_cast<const char *>(ptrEnergy.get()),
                 count * sizeof(basic::GlobalSingle_t::typeof_energy));
    break;
  default:
    // Do nothing, as the header is invalid
    break;
  }
}

void single::SingleFileOutput_impl::write2BufferTimeValue(
    std::unique_ptr<openpni::basic::GlobalSingle::typeof_timeValue_pico[]> &ptrTimeValue,
    uint64_t count, openpni::common::ReuseableStream &buffer) {
  switch (TimeValueType(m_header.bytes4TimeValue)) {
  case TimeValueType::UINT32: {
    const auto cuttedTimeValue32 =
        common::cut_to_bytes<basic::GlobalSingle_t::typeof_timeValue_pico, 4>(
            ptrTimeValue.get(), ptrTimeValue.get() + count);
    buffer.write(cuttedTimeValue32.get(), count * 4);
  } break;
  case TimeValueType::UINT40: {
    const auto cuttedTimeValue40 =
        common::cut_to_bytes<basic::GlobalSingle_t::typeof_timeValue_pico, 5>(
            ptrTimeValue.get(), ptrTimeValue.get() + count);
    buffer.write(cuttedTimeValue40.get(), count * 5);
  } break;
  case TimeValueType::UINT48: {
    const auto cuttedTimeValue48 =
        common::cut_to_bytes<basic::GlobalSingle_t::typeof_timeValue_pico, 6>(
            ptrTimeValue.get(), ptrTimeValue.get() + count);
    buffer.write(cuttedTimeValue48.get(), count * 6);
  } break;
  case TimeValueType::UINT56: {
    const auto cuttedTimeValue56 =
        common::cut_to_bytes<basic::GlobalSingle_t::typeof_timeValue_pico, 7>(
            ptrTimeValue.get(), ptrTimeValue.get() + count);
    buffer.write(cuttedTimeValue56.get(), count * 7);
  } break;
  case TimeValueType::UINT64: {
    buffer.write(reinterpret_cast<const char *>(ptrTimeValue.get()),
                 count * sizeof(basic::GlobalSingle_t::typeof_timeValue_pico));
  } break;
  default:
    // Do nothing, as the header is invalid
    break;
  }
}

void single::SingleFileOutput_impl::write2BufferCrystalIndex(
    std::unique_ptr<openpni::basic::GlobalSingle::typeof_globalCrystalIndex[]>
        &ptrCrystalIndex,
    uint64_t count, openpni::common::ReuseableStream &buffer) {
  switch (CrystalIndexType(m_header.bytes4CrystalIndex)) {
  case CrystalIndexType::UINT16: {
    const auto cuttedCrystalIndex16 =
        common::cut_to_bytes<basic::GlobalSingle_t::typeof_globalCrystalIndex, 2>(
            ptrCrystalIndex.get(), ptrCrystalIndex.get() + count);
    buffer.write(cuttedCrystalIndex16.get(), count * 2);
  } break;
  case CrystalIndexType::UINT24: {
    const auto cuttedCrystalIndex24 =
        common::cut_to_bytes<basic::GlobalSingle_t::typeof_globalCrystalIndex, 3>(
            ptrCrystalIndex.get(), ptrCrystalIndex.get() + count);
    buffer.write(cuttedCrystalIndex24.get(), count * 3);
    break;
  }
  case CrystalIndexType::UINT32: {
    buffer.write(reinterpret_cast<const char *>(ptrCrystalIndex.get()),
                 count * sizeof(basic::GlobalSingle_t::typeof_globalCrystalIndex));
    break;
  }
  default:
    // Do nothing, as the header is invalid
    break;
  }
}

uint64_t
single::SingleFileOutput_impl::calculateSegmentBytes(uint64_t count) const noexcept {
  uint64_t bytes = sizeof(SingleSegmentHeader);
  bytes += count * m_header.bytes4CrystalIndex;
  bytes += count * m_header.bytes4TimeValue;
  bytes += count * m_header.bytes4Energy;
  return bytes;
}
} // namespace openpni::io::single