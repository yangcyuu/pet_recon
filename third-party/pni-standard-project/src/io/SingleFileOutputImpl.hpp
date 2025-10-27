#pragma once
#include <fstream>
#include <thread>

#include "include/io/SingleIO.hpp"
#include "include/misc/CycledBuffer.hpp"
#include "src/common/ReuseableStream.hpp"
namespace openpni::io::single {
class SingleFileOutput_impl {
public:
  SingleFileOutput_impl() noexcept;
  ~SingleFileOutput_impl() noexcept;

  bool setBytes4CrystalIndex(CrystalIndexType type) noexcept;
  bool setBytes4TimeValue(TimeValueType type) noexcept;
  bool setBytes4Energy(EnergyType type) noexcept;
  void setTotalCrystalNum(uint32_t num) noexcept;
  void setReservedBytes(uint64_t r) noexcept;

public:
  void open(std::string path);
  bool appendSegment(const basic::GlobalSingle_t *data, uint64_t count, uint64_t clock, uint32_t duration) noexcept;

  void write2BufferEnergy(uint64_t count, std::unique_ptr<openpni::basic::GlobalSingle::typeof_energy[]> &ptrEnergy,
                          openpni::common::ReuseableStream &buffer);

  void write2BufferTimeValue(std::unique_ptr<openpni::basic::GlobalSingle::typeof_timeValue_pico[]> &ptrTimeValue,
                             uint64_t count, openpni::common::ReuseableStream &buffer);

private:
  void
  write2BufferCrystalIndex(std::unique_ptr<openpni::basic::GlobalSingle::typeof_globalCrystalIndex[]> &ptrCrystalIndex,
                           uint64_t count, openpni::common::ReuseableStream &buffer);
  uint64_t calculateSegmentBytes(uint64_t count) const noexcept;

private:
  struct InputBufferItem {
    std::vector<basic::GlobalSingle_t> data;
    std::size_t count{0};
    SingleSegmentHeader header;
  };

private:
  uint64_t m_reversedBytes{1024ull * 1024ull * 1024ull * 20ull}; // 默认保留空间为20GB
private:
  std::string m_filePath;
  std::ofstream m_file;
  SingleFileHeader m_header;
  std::thread m_internalOutputThread;
  std::thread m_internalCompressionThread;
  // 两个异步缓冲区
  common::CycledBuffer<InputBufferItem> m_inputBuffer{2};
  common::CycledBuffer<common::ReuseableStream> m_outputBuffer{2};
  std::atomic<uint64_t> m_bufferedBytes{0};
};
} // namespace openpni::io::single
