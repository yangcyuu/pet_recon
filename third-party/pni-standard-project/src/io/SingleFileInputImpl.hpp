#pragma once
#include <fstream>
#include <future>

#include "include/io/IO.hpp"
namespace openpni::io::single {
class SingleFileInput_impl {
public:
  SingleFileInput_impl() noexcept;
  ~SingleFileInput_impl() noexcept;

public:
  void open(std::string path);
  SingleFileHeader header() const noexcept;
  uint32_t segmentNum() const noexcept;
  SingleSegmentHeader segmentHeader(uint32_t segmentIndex) const noexcept;
  SingleSegmentBytes readSegment(uint32_t segmentIndex, uint32_t prefetchIndex) noexcept;

private:
  struct SingleSegmentInfo {
    SingleSegmentHeader header;
    size_t offsetOfCrystalIndex;
    size_t offsetOfTimeValue;
    size_t offsetOfEnergy;
  };
  using PrefetchType = std::pair<uint32_t, SingleSegmentBytes>;

private:
  SingleSegmentBytes readSegment_impl(uint32_t segmentIndex);

private:
  std::string m_filePath;
  std::ifstream m_file;
  uint64_t m_fileSize{0};
  SingleFileHeader m_header;
  std::vector<SingleSegmentInfo> m_segmentInfos;
  std::future<PrefetchType> m_prefetchFuture;
};
} // namespace openpni::io::single
