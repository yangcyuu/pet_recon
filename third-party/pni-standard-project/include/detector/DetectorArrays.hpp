// #pragma once
// #include <set>
// #include <string>
// #include <memory>
// #include <vector>
// #include <any>
// #include "../basic/PetDataType.h"
// #include "../basic/Math.hpp"
// #include "../PnI-Config.hpp"
// #include "../math/Geometry.hpp"
// #include "Detectors.hpp"

// #if !PNI_STANDARD_CONFIG_DISABLE_CUDA
// #include <cuda_runtime.h>
// #endif

// namespace openpni::device::deviceArray
// {
//     using namespace openpni::device;

//     // 基础参数结构体接口
//     struct BaseArrayR2SParams
//     {
//         virtual ~BaseArrayR2SParams() = default;
//         virtual std::unique_ptr<BaseArrayR2SParams> clone() const = 0;
//     };

//     class AbstractDetectorArray
//     {
//     public:
//         virtual ~AbstractDetectorArray() noexcept = default;

//         // Load detectors into the array
//         virtual bool loadDetector(std::vector<DetectorChangable> &DetectorChangable,
//                                   std::vector<std::string> &caliFiles) = 0;

//         virtual bool isDetectorLoaded() const noexcept = 0;

// #if !PNI_STANDARD_CONFIG_DISABLE_CUDA
//         virtual cudaError_t loadDetectorToGpu(std::vector<DetectorChangable> &DetectorChangable,
//                                               std::vector<std::string> &caliFiles,
//                                               const int deviceId) = 0;

//         virtual bool isDetectorLoadedToGpu() const noexcept = 0;

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
//                                      std::function<void(std::string)>()) const noexcept = 0;
// #endif

//         // Detector related information
//         virtual DetectorUnchangable detectorUnchangable() const noexcept = 0;
//         virtual DetectorChangable &detectorChangable(const uint64_t __detectorId) noexcept = 0;
//         virtual const DetectorChangable &detectorChangable(const uint64_t __detectorId) const noexcept = 0;
//     };
// }