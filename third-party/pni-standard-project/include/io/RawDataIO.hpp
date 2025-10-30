#pragma once
#include "../process/Acquisition.hpp"
namespace openpni::io::rawdata {
/** RawData文件格式：
 *  --> Header：512字节
 *  --> Detector Info：每个探测器的类型，总字节数为channelNum * 32字节
 *  --> Segment0：Segment Header0 -> [data]0 -> [channel]0 -> [length]0 -> [offset]0
 *  --> Segment1：Segment Header1 -> [data]1 -> [channel]1 -> [length]1 -> [offset]1
 *  --> Segment2：Segment Header2 -> [data]2 -> [channel]2 -> [length]2 -> [offset]2
 *  --> ......
 */
constexpr int RAWDATA_FILE_HEADER_SIZE = 512; // RawData文件头大小
#pragma pack(push, 1)
struct RawdataHeader {
  char fileTypeName[12]{"PNI-RAWDATA"};
  uint16_t version{0};
  uint32_t segmentNum{0};
  uint16_t channelNum{0};
};
#pragma pack(pop)

constexpr int RAWDATA_SEGMENT_HEADER_SIZE = 128; // RawData段头大小
#pragma pack(push, 1)
struct SegmentHeader {
  uint64_t count{0};
  uint64_t dataByteSize{0};
  uint64_t clock{0};
  uint32_t duration{0};
};
#pragma pack(pop)

struct RawdataSegment {
  std::unique_ptr<uint8_t[]> data;     // 存放数据的地方
  std::unique_ptr<uint16_t[]> length;  // 每个数据包的长度
  std::unique_ptr<uint64_t[]> offset;  // 每个数据包的起始字节偏移量
  std::unique_ptr<uint16_t[]> channel; // 每个数据包来自哪个通道
  openpni::process::RawDataView view(const RawdataHeader &, const SegmentHeader &) const;
};

class RawFileInputImpl;
class RawFileInput // 用于将RawData文件输入程序的类
{

public:
  RawFileInput() noexcept;
  ~RawFileInput() noexcept;

public:
  void open(std::string path);
  uint32_t segmentNum() const noexcept;
  RawdataHeader header() const noexcept;
  std::string typeNameOfChannel(uint16_t channelIndex) const noexcept;
  SegmentHeader segmentHeader(uint32_t segmentIndex) const noexcept;
  RawdataSegment readSegment(uint32_t segmentIndex, uint32_t prefetchIndex = uint32_t(-1)) const noexcept;

private:
  std::unique_ptr<RawFileInputImpl> impl;
};

class RawFileOutputImpl;
class RawFileOutput // 用于将RawData输出文件的类
{
public:
  RawFileOutput() noexcept;
  ~RawFileOutput() noexcept;

  void setChannelNum(uint16_t channelNum) noexcept;
  void setTypeNameOfChannel(uint16_t channelIndex, const std::string &typeName) noexcept;
  void setReservedBytes(uint64_t r) noexcept;

public:
  void open(std::string path);
  bool appendSegment(openpni::process::RawDataView rawData) noexcept;

private:
  std::unique_ptr<RawFileOutputImpl> impl;
};

} // namespace openpni::io::rawdata
