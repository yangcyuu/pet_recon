// #include "include/detector/BDM50100Array.hpp"
// #include "BDM50100Array_impl.hpp"

// namespace openpni::device::deviceArray
// {
//     // 构造函数
//     BDM50100Array::BDM50100Array() noexcept
//         : m_impl(std::make_unique<bdm50100Array::BDM50100Array_impl>())
//     {
//     }

//     // 析构函数
//     BDM50100Array::~BDM50100Array() noexcept
//     {
//         // 智能指针会自动清理资源
//     }

//     // 重写 AbstractDetectorArray 的虚函数
//     bool BDM50100Array::loadDetector(std::vector<DetectorChangable> &DetectorChangable,
//                                      std::vector<std::string> &caliFiles)
//     {
//         if (!m_impl)
//         {
//             return false;
//         }
//         return m_impl->loadDetector(DetectorChangable, caliFiles);
//     }

//     bool BDM50100Array::isDetectorLoaded() const noexcept
//     {
//         if (!m_impl)
//         {
//             return false;
//         }
//         return m_impl->isDetectorLoaded();
//     }

// #if !PNI_STANDARD_CONFIG_DISABLE_CUDA
//     cudaError_t BDM50100Array::loadDetectorToGpu(std::vector<DetectorChangable> &DetectorChangable,
//                                                  std::vector<std::string> &caliFiles,
//                                                  const int deviceId)
//     {
//         if (!m_impl)
//         {
//             return cudaErrorNotReady;
//         }
//         return m_impl->loadDetectorToGpu(DetectorChangable, caliFiles, deviceId);
//     }

//     bool BDM50100Array::isDetectorLoadedToGpu() const noexcept
//     {
//         if (!m_impl)
//         {
//             return false;
//         }
//         return m_impl->isDetectorLoadedToGpu();
//     }

//     cudaError_t BDM50100Array::r2s_cuda(const void *__d_raw,
//                                         const PacketPositionInfo *__d_position,
//                                         uint64_t __count,
//                                         basic::RigionalSingle_t *__d_out,
//                                         uint64_t &__outSinglesNum,
//                                         uint64_t &__bufferSize,
//                                         void *__d_buffer,
//                                         const BaseArrayR2SParams *__params,
//                                         cudaStream_t __stream,
//                                         std::function<void(std::string)> __callBackFunc) const noexcept
//     {
//         if (!m_impl)
//         {
//             return cudaErrorNotReady;
//         }

//         // 安全地转换参数类型
//         const BDM50100ArrayR2SParams *params = nullptr;
//         if (__params != nullptr)
//         {
//             params = dynamic_cast<const BDM50100ArrayR2SParams *>(__params);
//             if (params == nullptr)
//             {
//                 // 如果转换失败，说明传入的参数类型不正确
//                 if (__callBackFunc)
//                 {
//                     __callBackFunc("错误：传入的参数类型不匹配，期望 BDM50100ArrayR2SParams 类型");
//                 }
//                 return cudaErrorInvalidValue;
//             }
//         }

//         return m_impl->r2s_cuda(__d_raw,
//                                 __d_position,
//                                 __count,
//                                 __d_out,
//                                 __outSinglesNum,
//                                 __bufferSize,
//                                 __d_buffer,
//                                 params,
//                                 __stream,
//                                 __callBackFunc);
//     }
// #endif

//     // 探测器相关信息
//     DetectorUnchangable BDM50100Array::detectorUnchangable() const noexcept
//     {
//         if (!m_impl)
//         {
//             return DetectorUnchangable{};
//         }
//         return m_impl->detectorUnchangable();
//     }

//     DetectorChangable &BDM50100Array::detectorChangable(const uint64_t __detectorId) noexcept
//     {
//         if (!m_impl)
//         {
//             static DetectorChangable defaultDetector{};
//             return defaultDetector;
//         }
//         return m_impl->detectorChangable(__detectorId);
//     }

//     const DetectorChangable &BDM50100Array::detectorChangable(const uint64_t __detectorId) const noexcept
//     {
//         if (!m_impl)
//         {
//             static const DetectorChangable defaultDetector{};
//             return defaultDetector;
//         }
//         return m_impl->detectorChangable(__detectorId);
//     }

// }