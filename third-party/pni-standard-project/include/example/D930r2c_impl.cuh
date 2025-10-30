#pragma once
#include <cuda_runtime.h>
#include <memory>
#include <mutex>

#include "../basic/CudaUniquePtr.cuh"
#include "../detector/BDM50100Array.hpp"
#include "../r2cHandle/s2cAnyHandle.cuh"
// #include "../r2cHandle/s2cHandle.cuh"

namespace openpni::example::r2c_example::D930 {

using namespace openpni::basic;
using namespace openpni::device;
using namespace openpni::device::bdm50100;
using namespace openpni::device::deviceArray;
using namespace openpni::process::s2c_any;

typedef struct D930r2cParams {
  // 50100探测器校正算法参数
  bool matchXTalkEnabled = false;                     // 是否启用串扰匹配
  float timeWindow = 50.0f;                           // 时间窗口
  float timeShift = 75.0f;                            // 时间偏移
  bool crossTalkEnabled = false;                      // 是否启用串扰校正
  float crossTalkTimeWindow = 2.0f;                   // 串扰时间窗口
  DeviceModel deviceModel = DeviceModel::DIGITMI_930; // 设备型号
  caliCoef::EnergyThresholds_t energyThresholds = {60,  80, 100,    120,    140,   160,    180,
                                                   200, 0,  0.0454, 0.1111, 1.964, -0.0014}; // 能量阈值数组

  // 符合参数
  float energyLowKev = 421.0f;        // 能量低端
  float energyHighKev = 1000.0f;      // 能量高端
  uint64_t timeWindowPicosec = 2000;  // 时间窗口皮秒
  uint64_t delayTimePicosec = 100000; // 延迟时间皮秒
  uint32_t range = 13;                // 范围
} D930r2cParams_t;

namespace constants {
constexpr int32_t BDM_PER_RING = 48; // 每个环路的BDM数量
constexpr int32_t RING_NUM = 3;      // 环路数量
} // namespace constants

class D930r2c_impl final {

public:
  D930r2c_impl();
  D930r2c_impl(const D930r2c_impl &) = delete;
  D930r2c_impl(D930r2c_impl &&) = delete;
  D930r2c_impl &operator=(const D930r2c_impl &) = delete;
  D930r2c_impl &operator=(D930r2c_impl &&) = delete;
  ~D930r2c_impl();

public:
  cudaError_t init(const uint64_t maxFrameNum, const uint32_t deviceId,
                   std::vector<DetectorChangable> &DetectorChangable, std::vector<std::string> &caliFiles,
                   std::function<void(std::string)> logFunc = nullptr);

  cudaError_t excSync(const DataFrame50100_t *h_p_dataFrame, const PacketPositionInfo *h_p_packetPositionInfo,
                      const uint32_t frameNum, const D930r2cParams_t params, basic::Listmode_t *h_p_d_promptListmode,
                      basic::Listmode_t *h_p_d_delayListmode, uint32_t &promptNum, uint32_t &delayNum,
                      cudaStream_t stream = cudaStreamDefault, std::function<void(std::string)> logFunc = nullptr);

  cudaError_t getLastValidSingleAsync(basic::GlobalSingle_t *h_p_d_globalSingles, uint32_t &singlesNum,
                                      cudaStream_t stream = cudaStreamDefault,
                                      std::function<void(std::string)> logFunc = nullptr);

  cudaError_t getCryMap(std::vector<unsigned> &cryMap);

  cudaError_t resetCryMap();

  bool isAvailable() const noexcept;

  void lock() noexcept;

  bool tryLock() noexcept;

  void unlock() noexcept;

private:
  cudaError_t convertRigionalToGlobal(const basic::RigionalSingle_t *d_rigionalSingles,
                                      basic::GlobalSingle_t *d_globalSingles, const uint32_t singlesNum,
                                      cudaStream_t stream = cudaStreamDefault,
                                      std::function<void(std::string)> logFunc = nullptr);

private:
  // control
  bool m_isAvailable;
  uint32_t m_maxFrameNum;
  uint32_t m_deviceId;
  std::mutex m_mutex;
  size_t m_bufferSize;

  // buffer
  cuda::cuda_unique_ptr<DataFrame50100_t> m_d_dataFrame;
  cuda::cuda_unique_ptr<PacketPositionInfo> m_d_packetPositionInfo;
  cuda::cuda_unique_ptr<char> m_d_buffer;
  cuda::cuda_unique_ptr<basic::RigionalSingle_t> m_d_rigionalSingle;
  cuda::cuda_unique_ptr<basic::RigionalSingle_t> m_d_promptCoinSingles;
  cuda::cuda_unique_ptr<basic::RigionalSingle_t> m_d_delayCoinSingles;
  cuda::cuda_unique_ptr<basic::Listmode_t> m_d_promptListmode;
  cuda::cuda_unique_ptr<basic::Listmode_t> m_d_delayListmode;
  cuda::cuda_unique_ptr<unsigned int> m_d_cryMap;

  // results
  uint32_t m_validSinglesNum = 0;

  // algorithm
  BDM50100Array m_d_bdmArray;
  s2cAnyHandle m_d_s2cHandle;
};

} // namespace openpni::example::r2c_example::D930