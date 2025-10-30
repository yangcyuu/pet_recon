// #include "BDM50100_impl.hpp"
// #include <iostream>
// #include <stdexcept>
// #include <fstream>

// #if !PNI_STANDARD_CONFIG_DISABLE_CUDA
// #include <cuda_runtime.h>
// #endif

// namespace openpni::device::bdm50100
// {
//     // 构造函数
//     BDM50100Runtime_impl::BDM50100Runtime_impl()
//         : m_h_caliTable(std::make_unique<BDM50100CalibrtionTable>())
//     {
//         // 初始化校准表
//         m_h_caliTable->tdcEnergyCoefs = std::make_unique<std::array<caliCoef::TDCArray_S, BLOCK_NUM>>();
//         m_h_caliTable->tdcTimeCoefs = std::make_unique<std::array<caliCoef::TDCArray_S, CRYSTAL_NUM>>();
//         m_h_caliTable->energyCoefs = std::make_unique<std::array<caliCoef::EnergyCoef, CRYSTAL_NUM>>();
//         m_h_caliTable->timeCoefs = std::make_unique<std::array<caliCoef::TimeCoef, CRYSTAL_NUM>>();

//         // 初始化探测器变量为默认值
//         m_detectorChangable = DetectorChangable{};
//     }

//     // 加载探测器配置
//     bool BDM50100Runtime_impl::loadDetector(const DetectorChangable &DetectorChangable,
//                                             const std::string &caliFile)
//     {
//         m_detectorChangable = DetectorChangable;
//         std::fstream fin(caliFile, std::ios::in | std::ios::binary);
//         if (!fin.is_open())
//         {
//             std::cerr << "Failed to open calibration file: " << caliFile << std::endl;
//             return false;
//         }
//         try
//         {
//             // 读取TDC能量系数
//             fin.read(reinterpret_cast<char *>(m_h_caliTable->tdcEnergyCoefs->data()), BLOCK_NUM *
//             sizeof(caliCoef::TDCArray_S));
//             // 读取TDC时间系数
//             fin.read(reinterpret_cast<char *>(m_h_caliTable->tdcTimeCoefs->data()), CRYSTAL_NUM *
//             sizeof(caliCoef::TDCArray_S));
//             // 读取能量系数
//             fin.read(reinterpret_cast<char *>(m_h_caliTable->energyCoefs->data()), CRYSTAL_NUM *
//             sizeof(caliCoef::EnergyCoef));
//             // 读取时间系数
//             fin.read(reinterpret_cast<char *>(m_h_caliTable->timeCoefs->data()), CRYSTAL_NUM *
//             sizeof(caliCoef::TimeCoef));

//             m_isCalibrationLoaded = true;
//         }
//         catch (const std::exception &e)
//         {
//             std::cerr << "Error reading calibration file: " << e.what() << std::endl;
//             return false;
//         }

//         return true;
//     }

//     // 获取探测器配置（非常量版本）
//     DetectorChangable &BDM50100Runtime_impl::detectorChangable() noexcept
//     {
//         return m_detectorChangable;
//     }

//     // 获取探测器配置（常量版本）
//     const DetectorChangable &BDM50100Runtime_impl::detectorChangable() const noexcept
//     {
//         return m_detectorChangable;
//     }

//     // 检查探测器是否已加载
//     bool BDM50100Runtime_impl::isDetectorLoaded() const noexcept
//     {
//         return m_isCalibrationLoaded;
//     }

// } // namespace openpni::device::bdm50100

// namespace openpni::device
// {
//     template <>
//     DetectorUnchangable detectorUnchangable<bdm50100::BDM50100Runtime>()
//     {
//         DetectorUnchangable result;
//         result.maxUDPPacketSize = bdm50100::MAX_UDP_PACKET_SIZE;
//         result.minUDPPacketSize = bdm50100::MIN_UDP_PACKET_SIZE;
//         result.maxSingleNumPerPacket = bdm50100::MAX_SINGLE_NUM_PER_PACKET;
//         result.minSingleNumPerPacket = bdm50100::MIN_SINGLE_NUM_PER_PACKET;
//         result.geometry.blockNumU = bdm50100::BLOCK_NUM_U;
//         result.geometry.blockNumV = bdm50100::BLOCK_NUM_V;

//         // TODO : Set the correct block and crystal sizes
//         // result.geometry.blockSizeU = bdm2::BLOCK_PITCH;
//         // result.geometry.blockSizeV = bdm2::BLOCK_PITCH;
//         // result.geometry.crystalNumU = bdm2::CRYSTAL_LINE;
//         // result.geometry.crystalNumV = bdm2::CRYSTAL_LINE;
//         // result.geometry.crystalSizeU = bdm2::CRYSTAL_SIZE;
//         // result.geometry.crystalSizeV = bdm2::CRYSTAL_SIZE;

//         return result;
//     }
// }
