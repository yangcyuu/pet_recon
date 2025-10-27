#pragma once
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "../PnI-Config.hpp"
#include "../basic/Math.hpp"
#include "../basic/PetDataType.h"
#include "../math/Geometry.hpp"
// Header: Device Abstract Layer

namespace openpni::device {
struct PacketPositionInfo {
  uint64_t offset;
  uint16_t length;
  uint16_t channel;
};
} // namespace openpni::device
namespace openpni::device {
namespace names {
constexpr unsigned TYPE_MAX_LENGTH = 32; // 探测器型号的最大长度
constexpr char BDM2[TYPE_MAX_LENGTH] = "BDM2";
constexpr char BDM1286[TYPE_MAX_LENGTH] = "BDM1286";
constexpr char BDM50100[TYPE_MAX_LENGTH] = "BDM50100"; // 用于当前930系统的探测器型号
constexpr char ANY[TYPE_MAX_LENGTH] = "ANY";           // 任意探测器型号
}; // namespace names

const inline std::set<std::string> setSupportedScanners = {
    names::BDM2,     // 用于小动物PET的探测器型号（名字有罗马字符，改写成数字）
    names::BDM1286,  // 用于临床与质子刀的探测器型号（名字暂定）
    names::BDM50100, // 用于930系统的探测器型号
    names::ANY       // 任意探测器型号
};

namespace bdm2 {
class BDM2Runtime;
};

namespace bdm50100 {
class BDM50100Runtime;
};

namespace customer {
class EmptyDetectorRuntime;
};

struct DetectorUnchangable {
  uint16_t maxUDPPacketSize;
  uint16_t minUDPPacketSize;
  uint16_t maxSingleNumPerPacket;
  uint16_t minSingleNumPerPacket;
  basic::DetectorGeometry geometry; // 探测器的几何信息
};
struct DetectorChangable {
  uint32_t ipSource;        // 源IP地址，为0表示ANY
  uint16_t portSource;      // 源端口号，为0表示ANY
  uint32_t ipDestination;   // 目的IP地址
  uint16_t portDestination; // 目的端口号
  basic::Coordinate3D<float> coordinate;
};

template <typename T>
DetectorUnchangable detectorUnchangable();
template <>
DetectorUnchangable detectorUnchangable<bdm2::BDM2Runtime>();
template <>
DetectorUnchangable detectorUnchangable<bdm50100::BDM50100Runtime>();

class DetectorBase {
public:
  DetectorBase() = default;
  virtual ~DetectorBase() = default;

public: // 探测器相关信息
  virtual DetectorUnchangable detectorUnchangable() const noexcept = 0;
  virtual DetectorChangable &detectorChangable() noexcept = 0;
  virtual const DetectorChangable &detectorChangable() const noexcept = 0;
  virtual const char *detectorType() const noexcept = 0;

public: // 运行时相关操作
  // Raw => Singles (CPU未完成)
  virtual void r2s_cpu() const noexcept = 0;
/**
 * @brief 将原始数据转化为单事件，CUDA版本代码
 * @param d_raw 原始数据，显存指针
 * @param d_position 原始数据的位置信息，显存指针
 * @param count 原始数据的packet数量
 * @param d_out 输出单事件数据，显存指针
 * @note 输出的LocalSingle_t数据必须保证：如果是无效的单事件，则crystalIndex均为0xffff
 * @note 可以保证d_out空间充足，至少为count * maxSingleNumPerPacket *
 * sizeof(LocalSingle_t)
 */
#if !PNI_STANDARD_CONFIG_DISABLE_CUDA
  virtual void r2s_cuda(const void *d_raw, const PacketPositionInfo *d_position, uint64_t count,
                        basic::LocalSingle_t *d_out) const noexcept = 0;
#endif

  /**
   * @brief
   * 加载校正表文件的函数。这是一个暂时的解决方案（更好的方案还没想出来）。文件格式为内部定义二进制格式。
   * @param filePath 校正表文件的路径
   * @note 如果文件格式不对，或者文件不存在，则会抛出异常
   */
  virtual void loadCalibration(std::string filePath) = 0;
  /**
   * @brief 检查校正表是否已加载
   */
  virtual bool isCalibrationLoaded() const noexcept = 0;
};

} // namespace openpni::device
