#pragma once
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>
namespace openpni::experimental::interface {
struct PacketsInfo {
  uint8_t const *raw;
  uint64_t const *offset;
  uint16_t const *length;
  uint16_t const *channel;
  uint64_t count;
};

#pragma pack(push, 1)
struct LocalSingle {
  unsigned short channelIndex;
  unsigned short crystalIndex;
  uint64_t timevalue_pico;
  float energy;
  __PNI_CUDA_MACRO__ static LocalSingle invalid() { return LocalSingle{0xFFFF, 0xFFFF, 0xFFFFFFFFFFFFFFFF, -1.0f}; }
};
#pragma pack(pop)

class SingleGenerator {
public:
  SingleGenerator() = default;
  virtual ~SingleGenerator() = default;

public:
  virtual void setChannelIndex(uint16_t channelIndex) noexcept = 0;
  /**
   * @brief
   * 加载校正表文件的函数。这是一个暂时的解决方案（更好的方案还没想出来）。文件格式为内部定义二进制格式。
   * @param filePath 校正表文件的路径
   * @note 如果文件格式不对，或者文件不存在，则会抛出异常
   */
  virtual void loadCalibration(std::string filePath) = 0;
  virtual bool isCalibrationLoaded() const noexcept = 0;

public: // 运行时相关操作
  /**
   * @brief 将原始数据转化为单事件，CUDA版本代码
   * @param raw 原始数据指针
   * @param position 原始数据的位置信息指针（范围）
   * @note 输出的单事件必须都是有效的事件
   */
  virtual std::span<LocalSingle const> r2s_cpu(PacketsInfo h_packets) const = 0;
  virtual std::span<LocalSingle const> r2s_cuda(PacketsInfo d_packets) const = 0;
};

class SingleGeneratorArray {
public:
  SingleGeneratorArray() = default;
  virtual ~SingleGeneratorArray() = default;

public:
  virtual bool addSingleGenerator(SingleGenerator *generator) noexcept = 0;
  virtual void clearSingleGenerators() noexcept = 0;

public:
  virtual std::span<LocalSingle const> r2s_cpu(PacketsInfo h_packets) const = 0;
  virtual std::span<LocalSingle const> r2s_cuda(PacketsInfo d_packets) const = 0;
};

} // namespace openpni::experimental::interface
