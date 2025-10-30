#pragma once

#include "../basic/PetDataType.h"

namespace openpni::io::single {
constexpr int SINGLE_FILE_HEADER_SIZE = 512;
#pragma pack(push, 1)
struct SingleFileHeader {
  char fileTypeName[12]{"PNI-SINGLE"};
  uint16_t version{0};
  uint32_t segmentNum{0};
  uint32_t cystalNum{0};         // 晶体数量
  uint8_t bytes4CrystalIndex{0}; // 每个晶体索引占用的字节数，正常范围在2～4之间
  uint8_t bytes4TimeValue{0};    // 每个时间值占用的字节数，正常范围在4～8之间
  uint8_t bytes4Energy{0};       // 每个能量值占用的字节数，正常范围为2或4或1（倍率4keV）或0（总是511）
};
#pragma pack(pop)

enum class CrystalIndexType : uint8_t {
  UINT16 = 2, // 2 bytes
  UINT24 = 3, // 3 bytes
  UINT32 = 4  // 4 bytes
};

enum class TimeValueType : uint8_t {
  UINT32 = 4, // 4 bytes
  UINT40 = 5, // 5 bytes
  UINT48 = 6, // 6 bytes
  UINT56 = 7, // 7 bytes
  UINT64 = 8  // 8 bytes
};

enum class EnergyType : uint8_t {
  ZERO = 0,   // 0 bytes，表示能量总是511keV
  UINT8 = 1,  // 1 byte，使用无符号8位整数，倍率4keV
  UFLT16 = 2, // 2 bytes，使用内置的无符号半精度浮点数
  FLT32 = 4,  // 4 bytes，使用IEEE 754单精度浮点数
};

enum class SingleFileFlagsInvalid : uint16_t {
  NONE = 0x00,                              // 无错误
  CRYSTAL_INDEX = 0x01,                     // 晶体索引
  TIME_VALUE = 0x02,                        // 时间值
  ENERGY = 0x04,                            // 能量值
  ALL = CRYSTAL_INDEX | TIME_VALUE | ENERGY // 所有内容
};

constexpr int SINGLE_SEGMENT_HEADER_SIZE = 128;
#pragma pack(push, 1)
struct SingleSegmentHeader {
  uint64_t count{0};    // 这一段数据的单事件个数
  uint64_t clock{0};    // 这一段数据的起始计算机时钟（单位：毫秒）
  uint32_t duration{0}; // 这一段数据的持续时间（单位：毫秒）
};
#pragma pack(pop)

struct SingleSegmentBytes {
  std::unique_ptr<char[]> crystalIndexBytes;
  std::unique_ptr<char[]> timeValueBytes;
  std::unique_ptr<char[]> energyBytes;
  uint64_t storagedBytesCrystalIndex{0};
  uint64_t storagedBytesTimeValue{0};
  uint64_t storagedBytesEnergy{0};
};

class SingleFileInput_impl;
class SingleFileInput {
public:
  SingleFileInput() noexcept;
  ~SingleFileInput() noexcept;

public:
  void open(std::string path);
  SingleFileHeader header() const noexcept;
  uint32_t segmentNum() const noexcept;
  SingleSegmentHeader segmentHeader(uint32_t segmentIndex) const noexcept;
  SingleSegmentBytes readSegment(uint32_t segmentIndex, uint32_t prefetchIndex = uint32_t(-1)) const noexcept;

private:
  std::unique_ptr<SingleFileInput_impl> impl;
};

class SingleFileOutput_impl;
class SingleFileOutput {
public:
  SingleFileOutput() noexcept;
  ~SingleFileOutput() noexcept;

  // 下面几个是必须配置的，如果不配置的话后面的存储会失败
  bool setBytes4CrystalIndex(CrystalIndexType type) noexcept;
  bool setBytes4TimeValue(TimeValueType type) noexcept;
  bool setBytes4Energy(EnergyType type) noexcept;
  void setTotalCrystalNum(uint32_t num) noexcept;
  void setReservedBytes(uint64_t r) noexcept;

public:
  void open(std::string path);
  bool appendSegment(const basic::GlobalSingle_t *data, uint64_t count, uint64_t clock, uint32_t duration);

private:
  std::unique_ptr<SingleFileOutput_impl> impl;
};
} // namespace openpni::io::single
