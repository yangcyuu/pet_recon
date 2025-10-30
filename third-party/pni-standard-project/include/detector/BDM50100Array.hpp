// #pragma once
// #include "BDM50100.hpp"
// #include "DetectorArrays.hpp"

// namespace openpni::device::deviceArray
// {
//     using namespace openpni::device::bdm50100;
//     namespace bdm50100Array
//     {
//         // BDM50100Array_impl is the implementation of BDM50100Array
//         class BDM50100Array_impl;
//     };

//     enum class DeviceModel : int
//     {
//         DIGITMI_920 = 0,
//         DIGITMI_925 = 1,
//         DIGITMI_930 = 2,
//         DIGITMI_930_24 = 3,
//         DIGITMI_i30 = 4,
//         PRIMEMI_020 = 5
//     };

//     // BDM50100专用的r2s_cuda参数结构体
//     struct BDM50100ArrayR2SParams : public openpni::device::deviceArray::BaseArrayR2SParams
//     {
//         // 算法参数
//         bool matchXTalkEnabled = false;   // 是否启用串扰匹配
//         float timeWindow = 50.0f;         // 时间窗口
//         float timeShift = 75.0f;          // 时间偏移
//         bool crossTalkEnabled = false;    // 是否启用串扰校正
//         float crossTalkTimeWindow = 2.0f; // 串扰时间窗口

//         DeviceModel deviceModel = DeviceModel::DIGITMI_930; // 设备型号
//         caliCoef::EnergyThresholds_t energyThresholds =
//             {60, 80, 100, 120, 140, 160, 180, 200, 0, 0.0454, 0.1111, 1.964, -0.0014}; // 能量阈值数组

//         // 克隆函数实现
//         virtual std::unique_ptr<openpni::device::deviceArray::BaseArrayR2SParams> clone() const override
//         {
//             return std::make_unique<BDM50100ArrayR2SParams>(*this);
//         }
//     };

//     constexpr uint16_t MAX_BDM_SUPPORTED_NUM = UINT16_MAX;

//     // BDM50100Array is a wrapper for BDM50100 detector array
//     class BDM50100Array : public AbstractDetectorArray
//     {
//     public:
//         BDM50100Array() noexcept;
//         virtual ~BDM50100Array() noexcept override;

//     public:
//         // 重写 AbstractDetectorArray 的虚函数
//         virtual bool loadDetector(std::vector<DetectorChangable> &DetectorChangable,
//                                   std::vector<std::string> &caliFiles) override;

//         virtual bool isDetectorLoaded() const noexcept override;

// #if !PNI_STANDARD_CONFIG_DISABLE_CUDA
//         virtual cudaError_t loadDetectorToGpu(std::vector<DetectorChangable> &DetectorChangable,
//                                               std::vector<std::string> &caliFiles,
//                                               const int deviceId) override;

//         virtual bool isDetectorLoadedToGpu() const noexcept override;

//         virtual cudaError_t r2s_cuda(const void *__d_raw,
//                                      const PacketPositionInfo *__d_position,
//                                      uint64_t __count,
//                                      basic::RigionalSingle_t *__d_out,
//                                      uint64_t &__outSinglesNum,
//                                      uint64_t &__bufferSize,
//                                      void *__d_buffer,
//                                      const BaseArrayR2SParams *__params,
//                                      cudaStream_t __stream = cudaStreamDefault,
//                                      std::function<void(std::string)> __callBackFunc =
//                                      std::function<void(std::string)>()) const noexcept override;
// #endif

//         // 探测器相关信息
//         virtual DetectorUnchangable detectorUnchangable() const noexcept override;
//         virtual DetectorChangable &detectorChangable(const uint64_t __detectorId) noexcept override;
//         virtual const DetectorChangable &detectorChangable(const uint64_t __detectorId) const noexcept override;

//     private:
//         std::unique_ptr<bdm50100Array::BDM50100Array_impl> m_impl; // BDM50100Array_impl 的实现
//     };
// }
