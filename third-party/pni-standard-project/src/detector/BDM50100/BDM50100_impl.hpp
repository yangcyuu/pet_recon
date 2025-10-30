// #pragma once
// #include "include/detector/Detectors.hpp"
// #include <array>
// #include <memory>
// #include "include/basic/Point.hpp"
// #include "include/detector/BDM50100.hpp"
// #include "src/autogen/autogen_xml.hpp"

// #if !PNI_STANDARD_CONFIG_DISABLE_CUDA
// #include "include/basic/CudaUniquePtr.cuh"
// #endif

// namespace openpni::device::bdm50100
// {
//     struct bdm50100CaliView
//     {
//         caliCoef::TDCArray_S *tdcEnergyCoefs; // TDC能量系数数组
//         caliCoef::TDCArray_S *tdcTimeCoefs;   // TDC时间系数数组
//         caliCoef::EnergyCoef *energyCoefs;    // 能量系数数组
//         caliCoef::TimeCoef *timeCoefs;        // 时间系数数组
//     };

//     class BDM50100Runtime_impl
//     {
//     public:
//         BDM50100Runtime_impl();
//         ~BDM50100Runtime_impl() = default;

//     public:
//         bool loadDetector(const DetectorChangable &DetectorChangable,
//                           const std::string &caliFile);

//         DetectorChangable &detectorChangable() noexcept;
//         const DetectorChangable &detectorChangable() const noexcept;

//         bool isDetectorLoaded() const noexcept;

// #if !PNI_STANDARD_CONFIG_DISABLE_CUDA
//         cudaError_t loadDetectorToGpu(const DetectorChangable &DetectorChangable,
//                                       const std::string &caliFile,
//                                       const int deviceId);

//         bool isDetectorLoadedToGpu() const noexcept;

//         bdm50100CaliView getCaliView() const noexcept;
// #endif

//     private:
//         std::unique_ptr<bdm50100::BDM50100CalibrtionTable> m_h_caliTable;
//         DetectorChangable m_detectorChangable;
//         bool m_isCalibrationLoaded{false};

// #if !PNI_STANDARD_CONFIG_DISABLE_CUDA
//         bool m_isDetectorLoadedToGpu{false};
//         int m_currentDeviceId{-1};

//         // TDC能量系数的CUDA指针, 理论长度是BLOCK_NUM * bdm50100::caliCoef::TDCArray_S::size()
//         basic::cuda::cuda_unique_ptr<caliCoef::TDCArray_S> m_d_tdcEnergyCoefs;

//         // TDC时间系数的CUDA指针, 理论长度是CRYSTAL_NUM * bdm50100::caliCoef::TDCArray_S::size()
//         basic::cuda::cuda_unique_ptr<caliCoef::TDCArray_S> m_d_tdcTimeCoefs;

//         // 能量系数的CUDA指针, 理论长度是CRYSTAL_NUM * sizeof(bdm50100::caliCoef::EnergyCoef)
//         basic::cuda::cuda_unique_ptr<caliCoef::EnergyCoef> m_d_energyCoefs;

//         // 时间系数的CUDA指针, 理论长度是CRYSTAL_NUM * sizeof(bdm50100::caliCoef::TimeCoef)
//         basic::cuda::cuda_unique_ptr<caliCoef::TimeCoef> m_d_timeCoefs;

// #endif
//     };
// }
