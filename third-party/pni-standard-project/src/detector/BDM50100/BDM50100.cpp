// #include "include/detector/BDM50100.hpp"
// #include "BDM50100_impl.hpp"

// namespace openpni::device::bdm50100
// {
//     BDM50100Runtime::BDM50100Runtime() noexcept
//     {
//         // m_impl = std::make_unique<bdm50100::BDM50100Runtime_impl>();
//     }

//     BDM50100Runtime::~BDM50100Runtime() noexcept
//     {
//         // 清理资源
//     }

//     void BDM50100Runtime::loadCalibration(const std::string filename)
//     {
//         // m_impl->loadCaliTable(filename);
//     }

//     bool BDM50100Runtime::isCalibrationLoaded() const noexcept
//     {
//         // return m_impl->getIsCalibrationLoaded();
//         return false; // Placeholder, implement actual logic
//     }

//     DetectorUnchangable BDM50100Runtime::detectorUnchangable() const noexcept
//     {
//         // return openpni::device::detectorUnchangable<bdm50100::BDM50100Runtime>();
//         DetectorUnchangable result;
//         return result;
//     }

//     DetectorChangable &BDM50100Runtime::detectorChangable() noexcept
//     {
//         // return m_impl->detectorChangable();
//         static DetectorChangable dummy;
//         return dummy; // Placeholder, implement actual logic
//     }

//     const DetectorChangable &BDM50100Runtime::detectorChangable() const noexcept
//     {
//         // return m_impl->detectorChangable();
//         static DetectorChangable dummy;
//         return dummy; // Placeholder, implement actual logic
//     }

//     void BDM50100Runtime::r2s_cpu() const noexcept
//     {
//         // CPU未完成
//         return;
//     }

// #ifndef PNI_STANDARD_CONFIG_DISABLE_CUDA
//     void BDM50100Runtime::r2s_cuda(const void *d_raw,
//                                    const PacketPositionInfo *d_position,
//                                    uint64_t count,
//                                    basic::LocalSingle_t *d_out) const noexcept
//     {
//         // CUDA版本代码未完成
//         return;
//     }
// #endif

// }