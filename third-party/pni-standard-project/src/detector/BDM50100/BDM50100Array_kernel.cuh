// #pragma once
// #include "include/detector/BDM50100.hpp"
// #include "include/detector/BDM50100Array.hpp"
// #include "include/example/Bdm50100ArrayLayout.cuh"
// #include "BDM50100_kernel.cuh"

// namespace openpni::device::deviceArray::bdm50100Array::arrayAlgo
// {
//     using namespace openpni::device::bdm50100;

//     template <algo::ResloveMode MODE>
//     __global__ void resolveGroupBDM50100Array_kernel(const char *d_buffer,
//                                                      const PacketPositionInfo *d_position,
//                                                      const std::size_t packetNum,
//                                                      caliCoef::TDCArray_S **d_tdcEnergyCalTable,
//                                                      caliCoef::TDCArray_S **d_tdcTimeCalTable,
//                                                      const caliCoef::EnergyThresholds_t energyThresholds,
//                                                      uint32_t *d_energyCountPrefix,
//                                                      caliCoef::EnergyEvent_t *d_energyEvent,
//                                                      uint32_t *d_timeCountPrefix,
//                                                      caliCoef::TimeEvent_t *d_timeEvent)
//     {
//         // 编译时检查模板参数的有效性
//         static_assert(MODE == algo::ResloveMode::isOnlyForFineTime || MODE == algo::ResloveMode::complete,
//                       "resolveGroupBDM50100_kernel only supports isOnlyForFineTime or complete mode. "
//                       "Use countGroupBDM50100_kernel for countOnly mode.");

//         int tid = blockDim.x * blockIdx.x + threadIdx.x;
//         if (tid >= packetNum)
//         {
//             return; // 超出范围，直接返回
//         }

//         char *d_packetBuffer = const_cast<char *>(d_buffer + tid * bdm50100::UDP_PACKET_SIZE);
//         PacketPositionInfo *posPtr = const_cast<PacketPositionInfo *>(d_position + tid);
//         caliCoef::EnergyEvent_t *d_energyEventStart = d_energyEvent + d_energyCountPrefix[tid];
//         caliCoef::TimeEvent_t *d_timeEventStart = d_timeEvent + d_timeCountPrefix[tid];
//         float *m_tdcEnergyCalTable = reinterpret_cast<float *>(d_tdcEnergyCalTable[posPtr->channel]);
//         float *m_tdcTimeCalTable = reinterpret_cast<float *>(d_tdcTimeCalTable[posPtr->channel]);

//         uint32_t indexEnergy = 0;
//         uint32_t indexTime = 0;

//         // 解析单个BDM50100数据包
//         algo::resolveOneBDM50100Package_device<MODE>(d_packetBuffer,
//                                                      posPtr,
//                                                      m_tdcEnergyCalTable,
//                                                      m_tdcTimeCalTable,
//                                                      energyThresholds,
//                                                      d_energyEventStart,
//                                                      d_timeEventStart,
//                                                      indexEnergy,
//                                                      indexTime);

//         return; // 解析成功，返回
//     }

//     __global__ void countGlobalBlockEvent_kernel(caliCoef::EnergyEvent_t *d_energyEvent,
//                                                  caliCoef::TimeEvent_t *d_timeEvent,
//                                                  uint32_t *energyCount,
//                                                  uint32_t *timeCount,
//                                                  const uint32_t energyEventNum,
//                                                  const uint32_t timeEventNum)
//     {
//         int tid = blockDim.x * blockIdx.x + threadIdx.x;

//         // 统计能量事件

//         if (tid < energyEventNum)
//         {
//             caliCoef::EnergyEvent_t event = d_energyEvent[tid];
//             uint32_t globalEventBlockId = algo::index_process::toGlobalBlockId(event.bdmId, event.channelId);
//             atomicAdd(&energyCount[globalEventBlockId], 1);
//         }

//         // 统计时间事件
//         if (tid < timeEventNum)
//         {
//             caliCoef::TimeEvent_t event = d_timeEvent[tid];
//             uint32_t globalEventBlockId = algo::index_process::toGlobalBlockId(event.bdmId, event.channelId);
//             atomicAdd(&timeCount[globalEventBlockId], 1);
//         }

//         return;
//     }

//     // 开启若干block，令每一个block处理一个晶体block内的能量事件和时间事件的匹配
//     __global__ void eventMatchArray_Kernel(caliCoef::EnergyEvent_t *energyEvent,
//                                            caliCoef::TimeEvent_t *timeEvent,
//                                            uint32_t *energyCountBlockPrefix,
//                                            uint32_t *timeCountBlockPrefix,
//                                            const bool allowXtalk,
//                                            const float tW,
//                                            const float tShift,
//                                            caliCoef::EnergyCoef_t **energyTable,
//                                            caliCoef::TimeCoef_t **timeTable,
//                                            caliCoef::RawGlobalSingle50100_CUDA_t *rawSingles)
//     {
//         const uint32_t blockId = blockIdx.x;
//         const uint32_t threadId = threadIdx.x;
//         const uint32_t blockEnergyStart = energyCountBlockPrefix[blockId];
//         const uint32_t blockTimeStart = timeCountBlockPrefix[blockId];
//         const uint32_t blockEnergyCount = energyCountBlockPrefix[blockId + 1] - energyCountBlockPrefix[blockId];
//         const uint32_t blockTimeCount = timeCountBlockPrefix[blockId + 1] - timeCountBlockPrefix[blockId];

//         // 为每个thread计算能量事件索引边界[start, end)
//         const uint32_t energyEventPerThread = blockEnergyCount / blockDim.x;
//         uint32_t energyIdStart = energyEventPerThread * threadId; // energyIdStart等价于singleStart
//         uint32_t energyIdEnd = energyEventPerThread * (threadId + 1);
//         if (energyIdEnd > blockEnergyCount)
//         {
//             energyIdEnd = blockEnergyCount; // 确保不越界
//         }

//         energyIdStart += blockEnergyStart; // 转换为全局索引
//         energyIdEnd += blockEnergyStart;

//         const uint32_t bdmIndex = algo::index_process::toGlobalBdmId(blockId);

//         algo::eventMatch_device(energyEvent + energyIdStart,
//                                 timeEvent + blockTimeStart,
//                                 energyIdEnd - energyIdStart,
//                                 blockTimeCount,
//                                 allowXtalk,
//                                 tW,
//                                 tShift,
//                                 energyTable[bdmIndex],
//                                 timeTable[bdmIndex],
//                                 rawSingles + energyIdStart);
//         return; // 匹配完成，返回
//     }

//     __global__ void xtalkCorrection_kernel(caliCoef::RawGlobalSingle50100_CUDA_t *d_tSortedSingles,
//                                            caliCoef::EnergyCoef_t **d_energyCoef,
//                                            const double xtalkTimeWindow,
//                                            const uint32_t forwardSearchSize, // 包括自身，搜索范围[0，Size-1]
//                                            const uint32_t rawGlobalSingleNum)
//     {
//         const uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
//         uint32_t startIndex = tid * forwardSearchSize;
//         uint32_t endIndex = startIndex + forwardSearchSize; // 搜索范围[startIndex, endIndex)
//         endIndex = min(endIndex, rawGlobalSingleNum);
//         if (startIndex >= rawGlobalSingleNum)
//         {
//             return; // 超出范围，直接返回
//         }

//         for (uint32_t curIndex = startIndex; curIndex < endIndex; ++curIndex)
//         {
//             if (d_tSortedSingles[curIndex].valid == 0)
//             {
//                 continue; // 如果当前单元无效，跳过
//             }

//             if (!d_tSortedSingles[curIndex].matchCount || d_tSortedSingles[curIndex].xtalkCount)
//             {
//                 continue; // 如果已经有串扰计数或没有匹配计数，跳过
//             }
//             uint32_t xtalkCount = 0;
//             // 默认rawGlobalSingle已经按时间排序
//             for (uint32_t anotherIndex = curIndex + 1; anotherIndex < endIndex; ++anotherIndex)
//             {
//                 if (d_tSortedSingles[anotherIndex].valid == 0)
//                 {
//                     continue; // 如果另一个单元无效，跳过
//                 }

//                 if (d_tSortedSingles[anotherIndex].timeValue_nano - d_tSortedSingles[curIndex].timeValue_nano >
//                     xtalkTimeWindow)
//                 {
//                     break;
//                 }

//                 bool valid = true;

//                 valid &= algo::index_process::isBlockAdjacent_device(
//                     d_tSortedSingles[curIndex].bdmId,
//                     d_tSortedSingles[anotherIndex].bdmId,
//                     d_tSortedSingles[curIndex].cryId,
//                     d_tSortedSingles[anotherIndex].cryId); // 判断两个单元是否在相邻block内

//                 valid &= (d_tSortedSingles[anotherIndex].matchCount > 0);  // 另一个单元必须有匹配计数
//                 valid &= (d_tSortedSingles[anotherIndex].xtalkCount == 0); // 另一个单元没有串扰计数

//                 if (valid)
//                 {
//                     d_tSortedSingles[anotherIndex].valid = 0;      // 标记为无效
//                     d_tSortedSingles[anotherIndex].isXtalk = 1;    // 标记为串扰
//                     d_tSortedSingles[anotherIndex].xtalkCount = 1; // 串扰计数加1

//                     const uint32_t bdmId = d_tSortedSingles[curIndex].bdmId;
//                     const uint32_t energyTableIndex = d_tSortedSingles[curIndex].cryId;
//                     float energy = d_tSortedSingles[anotherIndex].energy + d_tSortedSingles[curIndex].energy;

//                     if (d_energyCoef[bdmId][energyTableIndex].KXA > 0.0f)
//                     {
//                         energy = energy * d_energyCoef[bdmId][energyTableIndex].KXA +
//                                  d_energyCoef[bdmId][energyTableIndex].KXB;
//                     }
//                     d_tSortedSingles[curIndex].energy = energy;
//                     xtalkCount++;
//                 }
//                 else
//                 {
//                     /* code */
//                 }
//             }
//             // 更新串扰计数
//             d_tSortedSingles[curIndex].xtalkCount = xtalkCount;
//             if (xtalkCount > 0)
//             {
//                 d_tSortedSingles[curIndex].isXtalk = 1; // 标记为串扰
//             }
//         }

//         return;
//     }

//     // 选择有效的RawGlobalSingle数据，并转换为GlobalSingle格式
//     // 添加了array中全局晶体转换的计算
//     // constexpr deviceArray::DeviceModel MODEL = deviceArray::DeviceModel::DIGITMI_930;
//     template <deviceArray::DeviceModel MODEL>
//     __global__ void selectConvertRaw2Global_kernel(const caliCoef::RawGlobalSingle50100_CUDA_t *d_rawGlobalSingle,
//                                                    basic::GlobalSingle_t *d_selectedSingle,
//                                                    const uint32_t *d_prefixedMask,
//                                                    const uint32_t rawGlobalSingleNum)
//     {
//         int tid = blockDim.x * blockIdx.x + threadIdx.x;
//         // 检查线程ID是否在有效范围内
//         if (tid >= rawGlobalSingleNum)
//         {
//             return; // 超出范围，直接返回
//         }

//         // 根据前缀掩码选择有效的单事件数据
//         if (d_rawGlobalSingle[tid].valid)
//         {

//             // 将有效的RawGlobalSingle转换为GlobalSingle
//             basic::GlobalSingle_t temp;
//             switch (MODEL)
//             {
//             case deviceArray::DeviceModel::DIGITMI_930:
//             {
//                 temp.globalCrystalIndex = layout::D930::getGlobalCryId(d_rawGlobalSingle[tid].bdmId,
//                                                                        d_rawGlobalSingle[tid].cryId);
//                 break;
//             }
//             default:
//             {
//                 break;
//             }
//             }

//             temp.timeValue_pico = static_cast<uint64_t>(d_rawGlobalSingle[tid].timeValue_nano * 1000.0);
//             temp.energy = d_rawGlobalSingle[tid].energy;

//             d_selectedSingle[d_prefixedMask[tid]] = temp; // 使用前缀掩码索引存储选中的单事件数据
//         }
//         else
//         {
//         }

//         return;
//     }

//     __global__ void selectConvertRaw2Rigional_kernel(const caliCoef::RawGlobalSingle50100_CUDA_t *d_rawGlobalSingle,
//                                                      basic::RigionalSingle_t *d_selectedSingle,
//                                                      const uint32_t *d_prefixedMask,
//                                                      const uint32_t rawGlobalSingleNum)
//     {
//         int tid = blockDim.x * blockIdx.x + threadIdx.x;
//         // 检查线程ID是否在有效范围内
//         if (tid >= rawGlobalSingleNum)
//         {
//             return; // 超出范围，直接返回
//         }

//         // 根据前缀掩码选择有效的单事件数据
//         if (d_rawGlobalSingle[tid].valid)
//         {

//             // 将有效的RawGlobalSingle转换为GlobalSingle
//             basic::RigionalSingle_t temp;
//             temp.bdmIndex = d_rawGlobalSingle[tid].bdmId;
//             temp.crystalIndex = d_rawGlobalSingle[tid].cryId;
//             temp.timeValue_pico = static_cast<uint64_t>(d_rawGlobalSingle[tid].timeValue_nano * 1000.0);
//             temp.energy = d_rawGlobalSingle[tid].energy;

//             d_selectedSingle[d_prefixedMask[tid]] = temp; // 使用前缀掩码索引存储选中的单事件数据
//         }
//         else
//         {
//         }

//         return;
//     }

// }