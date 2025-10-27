#pragma once
#include <future>

#include "include/io/IO.hpp"
namespace openpni::io::listmode {
class ListmodeFileInput_impl {
public:
  ListmodeFileInput_impl() noexcept;
  ~ListmodeFileInput_impl() noexcept;

  void open(std::string path);
  uint32_t segmentNum() const noexcept;
  ListmodeFileHeader header() const noexcept;
  ListmodeSegmentHeader segmentHeader(uint32_t segmentIndex) const noexcept;
  ListmodeSegmentBytes readSegment(uint32_t segmentIndex, uint32_t prefetchIndex) noexcept;

private:
  struct SegmentInfo {
    ListmodeSegmentHeader header;
    uint64_t offsetOfCrystalIndex1{0};
    uint64_t offsetOfCrystalIndex2{0};
    uint64_t offsetOfTimeValue1_2{0};
  };
  using FutureType = std::pair<uint32_t, ListmodeSegmentBytes>;

private:
  ListmodeSegmentBytes readSegment_impl(uint32_t segmentIndex);

private:
  std::string m_filePath;
  std::ifstream m_file;
  ListmodeFileHeader m_header;
  uint64_t m_fileSize{0};
  std::vector<SegmentInfo> m_segmentInfos;
  std::future<FutureType> m_prefetchFuture;
};
} // namespace  openpni::io::listmode
