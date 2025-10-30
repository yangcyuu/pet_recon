// #include <iostream>

// #include "BDM50100_impl.hpp"
// #include "include/PnI-Config.hpp"
// #if !PNI_STANDARD_CONFIG_DISABLE_CUDA

// namespace openpni::device::bdm50100
// {
//     // 将探测器数据加载到GPU
//     cudaError_t BDM50100Runtime_impl::loadDetectorToGpu(const DetectorChangable &DetectorChangable,
//                                                         const std::string &caliFile,
//                                                         const int deviceId)
//     {
//         cudaError_t err = cudaSuccess;
//         int originalDeviceId = 0;

//         // 保存当前设备ID并切换到目标设备
//         err = cudaGetDevice(&originalDeviceId);
//         if (err != cudaSuccess)
//             return err;

//         err = cudaSetDevice(deviceId);
//         if (err != cudaSuccess)
//             return err;

//         try
//         {
//             // 首先在CPU上加载数据
//             if (!loadDetector(DetectorChangable, caliFile))
//             {
//                 cudaSetDevice(originalDeviceId);
//                 return cudaErrorInvalidValue;
//             }

//             // 分配GPU内存
//             m_d_tdcEnergyCoefs = basic::cuda::make_cuda_unique_ptr<caliCoef::TDCArray_S>(BLOCK_NUM);
//             m_d_tdcTimeCoefs = basic::cuda::make_cuda_unique_ptr<caliCoef::TDCArray_S>(CRYSTAL_NUM);
//             m_d_energyCoefs = basic::cuda::make_cuda_unique_ptr<caliCoef::EnergyCoef>(CRYSTAL_NUM);
//             m_d_timeCoefs = basic::cuda::make_cuda_unique_ptr<caliCoef::TimeCoef>(CRYSTAL_NUM);

//             // 将校准数据复制到GPU
//             err = cudaMemcpy(m_d_tdcEnergyCoefs.get(),
//                              m_h_caliTable->tdcEnergyCoefs->data(),
//                              BLOCK_NUM * sizeof(caliCoef::TDCArray_S),
//                              cudaMemcpyHostToDevice);
//             if (err != cudaSuccess)
//             {
//                 cudaSetDevice(originalDeviceId);
//                 return err;
//             }

//             err = cudaMemcpy(m_d_tdcTimeCoefs.get(),
//                              m_h_caliTable->tdcTimeCoefs->data(),
//                              CRYSTAL_NUM * sizeof(caliCoef::TDCArray_S),
//                              cudaMemcpyHostToDevice);
//             if (err != cudaSuccess)
//             {
//                 cudaSetDevice(originalDeviceId);
//                 return err;
//             }

//             err = cudaMemcpy(m_d_energyCoefs.get(),
//                              m_h_caliTable->energyCoefs->data(),
//                              CRYSTAL_NUM * sizeof(caliCoef::EnergyCoef),
//                              cudaMemcpyHostToDevice);
//             if (err != cudaSuccess)
//             {
//                 cudaSetDevice(originalDeviceId);
//                 return err;
//             }

//             err = cudaMemcpy(m_d_timeCoefs.get(),
//                              m_h_caliTable->timeCoefs->data(),
//                              CRYSTAL_NUM * sizeof(caliCoef::TimeCoef),
//                              cudaMemcpyHostToDevice);
//             if (err != cudaSuccess)
//             {
//                 cudaSetDevice(originalDeviceId);
//                 return err;
//             }

//             m_isDetectorLoadedToGpu = true;
//             m_currentDeviceId = deviceId;
//         }
//         catch (const std::exception &e)
//         {
//             std::cerr << "Failed to load detector to GPU: " << e.what() << std::endl;
//             err = cudaErrorUnknown;
//         }

//         // 恢复原设备ID
//         cudaSetDevice(originalDeviceId);
//         return err;
//     }

//     // 检查探测器是否已加载到GPU
//     bool BDM50100Runtime_impl::isDetectorLoadedToGpu() const noexcept
//     {
//         return m_isDetectorLoadedToGpu;
//     }

//     // 获取校准数据视图（用于CUDA内核）
//     bdm50100CaliView BDM50100Runtime_impl::getCaliView() const noexcept
//     {
//         bdm50100CaliView view;
//         view.tdcEnergyCoefs = m_d_tdcEnergyCoefs.get();
//         view.tdcTimeCoefs = m_d_tdcTimeCoefs.get();
//         view.energyCoefs = m_d_energyCoefs.get();
//         view.timeCoefs = m_d_timeCoefs.get();
//         return view;
//     }
// }

// #endif