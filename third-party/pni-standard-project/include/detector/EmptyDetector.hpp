#pragma once
#include "Detectors.hpp"

namespace openpni::device {
namespace any {

class EmptyDetectorRuntime final : public DetectorBase {
public:
  EmptyDetectorRuntime() noexcept {}
  virtual ~EmptyDetectorRuntime() noexcept {}

public: // 探测器相关信息
  virtual DetectorUnchangable detectorUnchangable() const noexcept { return m_detectorUnchangable; };
  virtual DetectorChangable &detectorChangable() noexcept { return m_detectorChangable; };
  virtual const DetectorChangable &detectorChangable() const noexcept { return m_detectorChangable; };
  virtual const char *detectorType() const noexcept { return "ANY"; };

public: // 运行时相关操作
  virtual void r2s_cpu() const noexcept {};
#if !PNI_STANDARD_CONFIG_DISABLE_CUDA
  virtual void r2s_cuda(const void *d_raw, const PacketPositionInfo *d_position, uint64_t count,
                        basic::LocalSingle_t *d_out) const noexcept {};
#endif
  virtual void loadCalibration(std::string filePath) {};
  virtual bool isCalibrationLoaded() const noexcept { return false; };

private:
  DetectorUnchangable m_detectorUnchangable;
  DetectorChangable m_detectorChangable;
};

} // namespace any
} // namespace openpni::device
