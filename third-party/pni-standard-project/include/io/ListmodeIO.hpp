#pragma once
#include <memory>

#include "../basic/PetDataType.h"
#include "SingleIO.hpp"
namespace openpni::io::listmode {
constexpr int LISTMODE_FILE_HEADER_SIZE = 512;
#pragma pack(push, 1)
struct ListmodeFileHeader {
  char fileTypeName[12]{"PNI-LSTMODE"};
  uint16_t version{0};
  uint32_t segmentNum{0};
  uint32_t cystalNum{0};          // 晶体数量
  uint8_t bytes4CrystalIndex1{0}; // 每个晶体索引占用的字节数，正常范围在2～4之间
  uint8_t bytes4CrystalIndex2{0}; // 每个晶体索引占用的字节数，正常范围在2～4之间
  uint8_t bytes4TimeValue1_2{
      0}; // 每个时间值占用的字节数，正常范围在0~2之间，若为0则表示时间差为0，若为1则表示时间差为value
          // * 256 皮秒，若为2则表示时间差为value * 1 皮秒
};
#pragma pack(pop)

using CrystalIndexType = single::CrystalIndexType; // 晶体索引类型

enum class TimeValue1_2Type : uint8_t {
  ZERO = 0, // 0 bytes, 表示时间差为0
  INT8 = 1, // 1 byte, 表示时间差为value * 256皮秒
  INT16 = 2 // 2 bytes, 表示时间差为value * 1皮秒
};

enum class ListmodeFileFlagsInvalid : uint16_t {
  NONE = 0x00,                     // 无错误
  CrystalIndexType1Invalid = 0x01, // 晶体索引类型1无效
  CrystalIndexType2Invalid = 0x02, // 晶体索引类型2无效
  TimeValue1_2TypeInvalid = 0x04,  // 时间值类型1_2无效
};

constexpr int LISTMODE_SEGMENT_HEADER_SIZE = 128;
#pragma pack(push, 1)
struct ListmodeSegmentHeader {
  uint64_t count{0};    // 这一段数据的单事件个数
  uint64_t clock{0};    // 这一段数据的起始计算机时钟（单位：毫秒）
  uint32_t duration{0}; // 这一段数据的持续时间（单位：毫秒）
};
#pragma pack(pop)

struct ListmodeSegmentBytes {
  std::unique_ptr<char[]> crystalIndex1Bytes; // 晶体索引1
  std::unique_ptr<char[]> crystalIndex2Bytes; // 晶体索引2
  std::unique_ptr<char[]> timeValue1_2Bytes;  // 时间值1 - 2
  uint64_t storagedBytesCrystalIndex1{0};
  uint64_t storagedBytesCrystalIndex2{0};
  uint64_t storagedBytesTimeValue1_2{0};
};

class ListmodeFileInput_impl;
class ListmodeFileInput {
public:
  ListmodeFileInput() noexcept;
  ~ListmodeFileInput() noexcept;

public:
  void open(std::string path);
  uint32_t segmentNum() const noexcept;
  ListmodeFileHeader header() const noexcept;
  ListmodeSegmentHeader segmentHeader(uint32_t segmentIndex) const noexcept;
  ListmodeSegmentBytes readSegment(uint32_t segmentIndex, uint32_t prefetchIndex = uint32_t(-1)) const noexcept;

private:
  std::unique_ptr<ListmodeFileInput_impl> impl;
};

class ListmodeFileOutput_impl;
class ListmodeFileOutput {
public:
  ListmodeFileOutput() noexcept;
  ~ListmodeFileOutput() noexcept;

  bool setBytes4CrystalIndex1(CrystalIndexType type) noexcept;
  bool setBytes4CrystalIndex2(CrystalIndexType type) noexcept;
  bool setBytes4CrystalIndex(
      CrystalIndexType type) noexcept {
    return setBytes4CrystalIndex1(type) && setBytes4CrystalIndex2(type);
  }
  bool setBytes4TimeValue1_2(TimeValue1_2Type type) noexcept;
  void setTotalCrystalNum(uint32_t num) noexcept;
  void setReservedBytes(uint64_t r) noexcept;

public:
  void open(std::string path);
  bool appendSegment(const openpni::basic::Listmode_t *data, uint64_t count, uint64_t clock,
                     uint32_t duration) noexcept;

private:
  std::unique_ptr<ListmodeFileOutput_impl> impl;
};
} // namespace openpni::io::listmode
