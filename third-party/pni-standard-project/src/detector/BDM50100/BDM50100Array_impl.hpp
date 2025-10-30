// #pragma once
// #include "BDM50100_impl.hpp"
// #include "include/detector/BDM50100Array.hpp"
// #include <vector>
// #include <memory>

// namespace openpni::device::deviceArray::bdm50100Array
// {
//     using namespace openpni::device::bdm50100;

//     class BDM50100Array_impl
//     {
//     public:
//         BDM50100Array_impl();
//         ~BDM50100Array_impl();

//     public:
//         // 加载探测器
//         bool loadDetector(std::vector<DetectorChangable> &DetectorChangable,
//                           std::vector<std::string> &caliFiles);

//         bool isDetectorLoaded() const noexcept;

// #if !PNI_STANDARD_CONFIG_DISABLE_CUDA
//         cudaError_t loadDetectorToGpu(std::vector<DetectorChangable> &DetectorChangable,
//                                       std::vector<std::string> &caliFiles,
//                                       const int deviceId);

//         bool isDetectorLoadedToGpu() const noexcept;

//         cudaError_t r2s_cuda(const void *__d_raw,
//                              const PacketPositionInfo *__d_position,
//                              uint64_t __count,
//                              basic::RigionalSingle_t *__d_out,
//                              uint64_t &__outSinglesNum,
//                              uint64_t &__bufferSize,
//                              void *__d_buffer,
//                              const BDM50100ArrayR2SParams *__params,
//                              cudaStream_t __stream,
//                              std::function<void(std::string)> __callBackFunc) const noexcept;
// #endif

//         // 探测器信息
//         DetectorUnchangable detectorUnchangable() const noexcept;
//         DetectorChangable &detectorChangable(const uint64_t __detectorId) noexcept;
//         const DetectorChangable &detectorChangable(const uint64_t __detectorId) const noexcept;

//     private:
// #if !PNI_STANDARD_CONFIG_DISABLE_CUDA
//         static std::size_t getBufferSizeNeeded(const uint64_t __count, const uint64_t __bdmNum) noexcept;

//         static cudaError_t selectConvertRaw2Global(const caliCoef::RawGlobalSingle50100_CUDA_t *d_rawGlobalSingle,
//                                                    basic::GlobalSingle_t *d_selectedSingle,
//                                                    const uint32_t *d_prefixedMask,
//                                                    const uint32_t rawGlobalSingleNum,
//                                                    const BDM50100ArrayR2SParams *__params,
//                                                    cudaStream_t __stream = 0,
//                                                    std::function<void(std::string)> __callBackFunc =
//                                                    std::function<void(std::string)>());

//         static cudaError_t selectConvertRaw2Rigional(const caliCoef::RawGlobalSingle50100_CUDA_t *d_rawGlobalSingle,
//                                                      basic::RigionalSingle_t *d_selectedSingle,
//                                                      const uint32_t *d_prefixedMask,
//                                                      const uint32_t rawGlobalSingleNum,
//                                                      cudaStream_t __stream = 0,
//                                                      std::function<void(std::string)> __callBackFunc =
//                                                      std::function<void(std::string)>());
// #endif

//     private:
//         bool m_isDetectorLoaded{false};
//         bool m_isDetectorLoadedToGpu{false};
//         std::vector<DetectorChangable> m_detectorChangable;
//         DetectorUnchangable m_detectorUnchangable;
//         int m_currentDeviceId{-1};

//         // BDM50100Runtime 实现的向量
//         std::vector<std::unique_ptr<openpni::device::bdm50100::BDM50100Runtime_impl>> m_detectorImpls;

// #if !PNI_STANDARD_CONFIG_DISABLE_CUDA
//         basic::cuda::cuda_unique_ptr<caliCoef::TDCArray_S *> m_d_tdcEnergyCoefsView;
//         basic::cuda::cuda_unique_ptr<caliCoef::TDCArray_S *> m_d_tdcTimeCoefsView;
//         basic::cuda::cuda_unique_ptr<caliCoef::EnergyCoef *> m_d_energyCoefsView;
//         basic::cuda::cuda_unique_ptr<caliCoef::TimeCoef *> m_d_timeCoefsView;
// #endif

//     public:
// #if !PNI_STANDARD_CONFIG_DISABLE_CUDA
//         static void test();
// #endif
//     };
// }