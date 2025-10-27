#pragma once
#include <thread>

#include "include/io/IO.hpp"
#include "include/misc/CycledBuffer.hpp"
#include "src/common/ReuseableStream.hpp"
namespace openpni::io::listmode {
class ListmodeFileOutput_impl {
public:
  ListmodeFileOutput_impl() noexcept;
  ~ListmodeFileOutput_impl() noexcept;

  bool setBytes4CrystalIndex1(CrystalIndexType type) noexcept;
  bool setBytes4CrystalIndex2(CrystalIndexType type) noexcept;
  bool setBytes4TimeValue1_2(TimeValue1_2Type type) noexcept;
  void open(std::string path);
  bool appendSegment(const basic::Listmode_t *data, uint64_t count, uint64_t clock, uint32_t duration) noexcept;
  void setTotalCrystalNum(
      uint32_t num) noexcept {
    m_header.cystalNum = num;
  }
  void setReservedBytes(
      uint64_t r) noexcept {
    m_reservedBytes = r;
  }

private:
  struct InputBufferItem {
    std::vector<basic::Listmode_t> data;
    std::size_t count{0};
    ListmodeSegmentHeader header;
  };
  void writeCrystalIndex2Buffer(openpni::basic::Listmode_t::typeof_globalCrystalIndex *ptrCrystalIndex, uint64_t count,
                                common::ReuseableStream &buffer, uint8_t bytes4EachElement) const noexcept;
  void writeTimeValue1_2ToBuffer(openpni::basic::Listmode_t::typeof_timeValue1_2 *ptrTimeValue, uint64_t count,
                                 common::ReuseableStream &buffer) const noexcept;
  uint64_t calculateSegmentBytes(uint64_t count) const noexcept;

private:
  std::string m_filePath;
  std::ofstream m_file;
  ListmodeFileHeader m_header;
  std::thread m_compressionThread;
  std::thread m_internalOutputThread;
  common::CycledBuffer<InputBufferItem> m_inputBuffer{2};
  common::CycledBuffer<common::ReuseableStream> m_outputBuffer{2};
  uint64_t m_reservedBytes{1024ull * 1024ull * 1024ull * 20ull}; // 默认保留空间为20GB
  std::atomic<uint64_t> m_bufferedBytes{0};
};
} // namespace openpni::io::listmode
