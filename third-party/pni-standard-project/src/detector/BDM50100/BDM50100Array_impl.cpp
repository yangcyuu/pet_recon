// #include "BDM50100Array_impl.hpp"
// #include <iostream>
// #include <stdexcept>

// namespace openpni::device::deviceArray::bdm50100Array
// {
//     // 构造函数
//     BDM50100Array_impl::BDM50100Array_impl()
//     {
//         // 初始化探测器几何信息
//         m_detectorUnchangable.maxUDPPacketSize = openpni::device::bdm50100::MAX_UDP_PACKET_SIZE;
//         m_detectorUnchangable.minUDPPacketSize = openpni::device::bdm50100::MIN_UDP_PACKET_SIZE;
//         m_detectorUnchangable.maxSingleNumPerPacket = openpni::device::bdm50100::MAX_SINGLE_NUM_PER_PACKET;
//         m_detectorUnchangable.minSingleNumPerPacket = openpni::device::bdm50100::MIN_SINGLE_NUM_PER_PACKET;
//         m_detectorUnchangable.geometry.blockNumU = openpni::device::bdm50100::BLOCK_NUM_U;
//         m_detectorUnchangable.geometry.blockNumV = openpni::device::bdm50100::BLOCK_NUM_V;

//         // TODO: 设置正确的块和晶体尺寸
//         // m_detectorUnchangable.geometry.blockSizeU = openpni::device::bdm50100::BLOCK_PITCH;
//         // m_detectorUnchangable.geometry.blockSizeV = openpni::device::bdm50100::BLOCK_PITCH;
//         // m_detectorUnchangable.geometry.crystalNumU = openpni::device::bdm50100::CRYSTAL_LINE;
//         // m_detectorUnchangable.geometry.crystalNumV = openpni::device::bdm50100::CRYSTAL_LINE;
//         // m_detectorUnchangable.geometry.crystalSizeU = openpni::device::bdm50100::CRYSTAL_SIZE;
//         // m_detectorUnchangable.geometry.crystalSizeV = openpni::device::bdm50100::CRYSTAL_SIZE;
//     }

//     // 析构函数
//     BDM50100Array_impl::~BDM50100Array_impl()
//     {
//         // 智能指针会自动清理资源
//     }

//     // 加载探测器阵列
//     bool BDM50100Array_impl::loadDetector(std::vector<DetectorChangable> &DetectorChangable,
//                                           std::vector<std::string> &caliFiles)
//     {
//         try
//         {
//             if (DetectorChangable.size() != caliFiles.size())
//             {
//                 std::cerr << "Error: DetectorChangable and caliFiles size mismatch" << std::endl;
//                 return false;
//             }

//             // 清理现有数据
//             m_detectorImpls.clear();
//             m_detectorChangable.clear();

//             // 为每个探测器创建实现
//             for (size_t i = 0; i < DetectorChangable.size(); ++i)
//             {
//                 auto impl = std::make_unique<openpni::device::bdm50100::BDM50100Runtime_impl>();

//                 if (!impl->loadDetector(DetectorChangable[i], caliFiles[i]))
//                 {
//                     std::cerr << "Failed to load detector " << i << std::endl;
//                     return false;
//                 }

//                 m_detectorImpls.push_back(std::move(impl));
//                 m_detectorChangable.push_back(DetectorChangable[i]);
//             }

//             m_isDetectorLoaded = true;
//             return true;
//         }
//         catch (const std::exception &e)
//         {
//             std::cerr << "Exception in loadDetector: " << e.what() << std::endl;
//             return false;
//         }
//     }

//     // 检查探测器是否已加载
//     bool BDM50100Array_impl::isDetectorLoaded() const noexcept
//     {
//         return m_isDetectorLoaded && !m_detectorImpls.empty();
//     }

//     // 获取探测器不变信息
//     DetectorUnchangable BDM50100Array_impl::detectorUnchangable() const noexcept
//     {
//         return m_detectorUnchangable;
//     }

//     // 获取探测器可变信息（非常量版本）
//     DetectorChangable &BDM50100Array_impl::detectorChangable(const uint64_t __detectorId) noexcept
//     {
//         if (__detectorId >= m_detectorChangable.size())
//         {
//             // 返回第一个探测器的配置作为默认值，如果没有探测器则创建默认配置
//             if (m_detectorChangable.empty())
//             {
//                 static DetectorChangable defaultDetector{};
//                 return defaultDetector;
//             }
//             return m_detectorChangable[0];
//         }
//         return m_detectorChangable[__detectorId];
//     }

//     // 获取探测器可变信息（常量版本）
//     const DetectorChangable &BDM50100Array_impl::detectorChangable(const uint64_t __detectorId) const noexcept
//     {
//         if (__detectorId >= m_detectorChangable.size())
//         {
//             // 返回第一个探测器的配置作为默认值，如果没有探测器则创建默认配置
//             if (m_detectorChangable.empty())
//             {
//                 static const DetectorChangable defaultDetector{};
//                 return defaultDetector;
//             }
//             return m_detectorChangable[0];
//         }
//         return m_detectorChangable[__detectorId];
//     }

// }