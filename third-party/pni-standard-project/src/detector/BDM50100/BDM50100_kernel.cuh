// #pragma once
// #include "include/detector/BDM50100.hpp"

// namespace openpni::device::bdm50100::algo
// {
//     namespace constant_number
//     {
//         constexpr double d2_3 = 8;
//         constexpr double d2_8 = 256;
//         constexpr double d2_16 = 65536;
//         constexpr double d2_24 = 16777216;
//         constexpr double d2_32 = 4294967296;
//         constexpr double d2_40 = 1099511627776;

//         // 在设备常量内存中定义邻接矩阵，避免nvcc编译问题
//         __device__ __constant__ int d_ADJACENCY_MATRIX[BLOCK_NUM][BLOCK_NUM] = {
//             {1, 1, 1, 1, 0, 0, 0, 0},
//             {1, 1, 1, 1, 0, 0, 0, 0},
//             {1, 1, 1, 1, 1, 1, 0, 0},
//             {1, 1, 1, 1, 1, 1, 0, 0},
//             {0, 0, 1, 1, 1, 1, 1, 1},
//             {0, 0, 1, 1, 1, 1, 1, 1},
//             {0, 0, 0, 0, 1, 1, 1, 1},
//             {0, 0, 0, 0, 1, 1, 1, 1}};
//     }

//     namespace index_process
//     {
//         // 实际channel（包含虚拟通道，[1,296]）至晶体编号[0,287]映射
//         __host__ __device__ uint16_t realChannelToCrystalId(const uint16_t realChannel)
//         {
//             return realChannel - bdm50100::MIN_CHANNEL_ID - realChannel / bdm50100::CHANNEL_NUM_PER_BLOCK;
//         }

//         // 实际channel（包含虚拟通道，[1,296]）至block编号[0,7]映射
//         __host__ __device__ uint16_t realChannelToBlockId(const uint16_t realChannel)
//         {
//             return (realChannel - bdm50100::MIN_CHANNEL_ID) / bdm50100::CHANNEL_NUM_PER_BLOCK;
//         }

//         // bdm编号([0, N-1])和通道编号([1, 296])至全局block编号映射
//         __host__ __device__ uint16_t toGlobalBlockId(const uint16_t bdmId, const uint16_t channelId)
//         {
//             return bdmId * bdm50100::BLOCK_NUM +
//                    (channelId - bdm50100::MIN_CHANNEL_ID) / bdm50100::CHANNEL_NUM_PER_BLOCK;
//         }

//         // 将全局block编号转换为bdm编号
//         __host__ __device__ uint16_t toGlobalBdmId(const uint16_t globalBlockId)
//         {
//             return globalBlockId / bdm50100::BLOCK_NUM;
//         }

//         __host__ __device__ uint32_t toGlobalCryId(const uint16_t &bdmId, const uint16_t &channelId)
//         {
//             uint32_t bdmId32 = static_cast<uint32_t>(bdmId);
//             uint32_t channelId32 = static_cast<uint32_t>(channelId);

//             return bdmId32 * bdm50100::CRYSTAL_NUM + channelId32 - bdm50100::MIN_CHANNEL_ID -
//                    channelId32 / bdm50100::CHANNEL_NUM_PER_BLOCK;
//         }

//         __host__ __device__ uint16_t toLocalBlockId(const uint16_t localCryId)
//         {
//             return localCryId / bdm50100::CRYSTAL_NUM_ONE_BLOCK;
//         }

//         __host__ __device__ int2 gCryIdToBdmCryId(const uint32_t globalCryId)
//         {
//             //(bdmid, cryid)
//             int2 bdmCryId;
//             bdmCryId.x = globalCryId / bdm50100::CRYSTAL_NUM;
//             bdmCryId.y = globalCryId % bdm50100::CRYSTAL_NUM;
//             return bdmCryId;
//         }

//         // 判断两个单事件是否在相邻block内
//         __device__ bool isBlockAdjacent_device(const uint16_t &bdmId_a,
//                                                const uint16_t &bdmId_b,
//                                                const uint16_t &cryId_a,
//                                                const uint16_t &cryId_b)
//         {
//             //
//             if (bdmId_a != bdmId_b)
//             {
//                 return false; // 不在同一个BDM内
//             }

//             const uint16_t blockId_a = cryId_a / bdm50100::CRYSTAL_NUM_ONE_BLOCK;
//             const uint16_t blockId_b = cryId_b / bdm50100::CRYSTAL_NUM_ONE_BLOCK;

//             if (blockId_a >= bdm50100::BLOCK_NUM || blockId_b >= bdm50100::BLOCK_NUM)
//             {
//                 return false; // 超出block范围
//             }

//             // 直接使用设备常量内存中的邻接矩阵
//             return constant_number::d_ADJACENCY_MATRIX[blockId_a][blockId_b] == 1;
//         }

//         typedef struct globalCryIdOpDefault
//         {
//             __host__ __device__ globalCryIdOpDefault() {}

//             // 将bdm编号（[0,N-1]）和通道编号（[1,296]）转换为全局晶体编号
//             __host__ __device__ __forceinline__ uint16_t operator()(const uint16_t &bdmId, const uint16_t &channelId)
//             const
//             {
//                 return bdmId * bdm50100::CRYSTAL_NUM + channelId - bdm50100::MIN_CHANNEL_ID -
//                        channelId / bdm50100::CHANNEL_NUM_PER_BLOCK;
//             }
//         } globalCryIdOpDefault_t;
//     }

//     namespace energy_algo
//     {
//         __device__ float CalcEnergy50100_device(const uint8_t len,
//                                                 const double KK_res_bits23,
//                                                 const double QQ_res_bits24,
//                                                 const double RR_res_bits42,
//                                                 caliCoef::EnergyThresholds_t energyThresholds)
//         {
//             double sumY = 0;
//             for (int i = 0; i < len / 2; i++)
//             {
//                 sumY += 2 * std::log(static_cast<double>(energyThresholds.data[i] / 1000));
//             }

//             uint32_t KK = static_cast<uint32_t>(KK_res_bits23 / (11 * 11));
//             int32_t QQ = -static_cast<int32_t>(QQ_res_bits24 / (11 * 11));
//             uint32_t RR = static_cast<uint32_t>(RR_res_bits42 / (11 * 11 * 11 * 11));

//             double PP = (QQ - sumY * KK / len) / (RR - KK * KK / len);

//             if (PP >= 0)
//             {
//                 return -1;
//             }

//             float energy = static_cast<float>(std::exp((sumY - PP * KK) / len) * std::sqrt(3.14) *
//                                               std::sqrt(-1 / PP));

//             if (energy != energy)
//             {
//                 return -1;
//             }

//             return energy;
//         }
//     }

//     namespace raw_frame_copy
//     {
//         __device__ TimePacket50100_t copyTimePacket(const DataFrame50100_t *frame, uint32_t offset)
//         {
//             TimePacket50100_t timePacketData;

//             // 检查边界，确保不会越界访问
//             if (frame == nullptr || offset + sizeof(TimePacket50100_t) > UDP_RAWDATA_SIZE)
//             {
//                 // 如果越界，返回零初始化的数据包
//                 timePacketData.EFCH1 = 0;
//                 timePacketData.EFCH2 = 0;
//                 timePacketData.T0B2 = 0;
//                 timePacketData.T0B3 = 0;
//                 timePacketData.T0B4 = 0;
//                 timePacketData.T0B5 = 0;
//                 timePacketData.T0B6 = 0;
//                 timePacketData.T0B7 = 0;
//                 timePacketData.T2H8 = 0;
//                 timePacketData.T2L8 = 0;
//                 return timePacketData;
//             }

//             // 使用安全的逐字节复制方式，避免直接使用memcpy
//             timePacketData.EFCH1 = frame->data[offset + 0];
//             timePacketData.EFCH2 = frame->data[offset + 1];
//             timePacketData.T0B2 = frame->data[offset + 2];
//             timePacketData.T0B3 = frame->data[offset + 3];
//             timePacketData.T0B4 = frame->data[offset + 4];
//             timePacketData.T0B5 = frame->data[offset + 5];
//             timePacketData.T0B6 = frame->data[offset + 6];
//             timePacketData.T0B7 = frame->data[offset + 7];
//             timePacketData.T2H8 = frame->data[offset + 8];
//             timePacketData.T2L8 = frame->data[offset + 9];

//             return timePacketData;
//         }

//         __device__ EnergyPacket50100_t copyEnergyPacket(const DataFrame50100_t *frame, uint32_t offset)
//         {
//             EnergyPacket50100_t energyPacketData;

//             // 检查边界，确保不会越界访问
//             if (frame == nullptr || offset + sizeof(EnergyPacket50100_t) > UDP_RAWDATA_SIZE)
//             {
//                 // 如果越界，返回零初始化的数据包
//                 energyPacketData.EFCH1 = 0;
//                 energyPacketData.EFCH2 = 0;
//                 energyPacketData.T0B2 = 0;
//                 energyPacketData.T0B3 = 0;
//                 energyPacketData.T0B4 = 0;
//                 energyPacketData.T0B5 = 0;
//                 energyPacketData.T0B6 = 0;
//                 energyPacketData.T0B7 = 0;
//                 energyPacketData.KK1 = 0;
//                 energyPacketData.KK2 = 0;
//                 energyPacketData.KK3 = 0;
//                 energyPacketData.QQ1 = 0;
//                 energyPacketData.QQ2 = 0;
//                 energyPacketData.QQ3 = 0;
//                 energyPacketData.RR1 = 0;
//                 energyPacketData.RR2 = 0;
//                 energyPacketData.RR3 = 0;
//                 energyPacketData.RR4 = 0;
//                 energyPacketData.RR5 = 0;
//                 energyPacketData.RR6 = 0;
//                 return energyPacketData;
//             }

//             // 使用安全的逐字节复制方式，避免直接使用memcpy
//             energyPacketData.EFCH1 = frame->data[offset + 0];
//             energyPacketData.EFCH2 = frame->data[offset + 1];
//             energyPacketData.T0B2 = frame->data[offset + 2];
//             energyPacketData.T0B3 = frame->data[offset + 3];
//             energyPacketData.T0B4 = frame->data[offset + 4];
//             energyPacketData.T0B5 = frame->data[offset + 5];
//             energyPacketData.T0B6 = frame->data[offset + 6];
//             energyPacketData.T0B7 = frame->data[offset + 7];
//             energyPacketData.KK1 = frame->data[offset + 8];
//             energyPacketData.KK2 = frame->data[offset + 9];
//             energyPacketData.KK3 = frame->data[offset + 10];
//             energyPacketData.QQ1 = frame->data[offset + 11];
//             energyPacketData.QQ2 = frame->data[offset + 12];
//             energyPacketData.QQ3 = frame->data[offset + 13];
//             energyPacketData.RR1 = frame->data[offset + 14];
//             energyPacketData.RR2 = frame->data[offset + 15];
//             energyPacketData.RR3 = frame->data[offset + 16];
//             energyPacketData.RR4 = frame->data[offset + 17];
//             energyPacketData.RR5 = frame->data[offset + 18];
//             energyPacketData.RR6 = frame->data[offset + 19];

//             return energyPacketData;
//         }

//     }

//     // 解析模式枚举
//     // countOnly: 只计数，不做解析
//     // isOnlyForFineTime: 只解析细时间，不计算能量
//     // complete: 完整解析，计算能量和细时间
//     enum class ResloveMode : int
//     {
//         countOnly = 0,
//         isOnlyForFineTime,
//         complete
//     };

//     /**
//      * @brief 解析单个BDM50100数据包，提取时间和能量事件
//      *
//      * 该模板函数根据指定的解析模式处理单个BDM50100数据包，将原始数据解析为
//      * 时间事件和能量事件。支持三种解析模式：仅计数、仅细时间、完整解析。
//      *
//      * @tparam MODE 解析模式枚举值
//      *              - countOnly: 仅统计事件数量，不进行实际数据解析
//      *              - isOnlyForFineTime: 仅解析细时间信息，不计算能量值
//      *              - complete: 完整解析，包括时间校正和能量计算
//      *
//      * @param d_buffer            指向GPU内存中单个数据包的指针
//      * @param posPtr              数据包位置信息结构体指针，包含BDM ID等信息
//      * @param tdcEnergyCalTable   TDC能量校正表，用于能量事件的精确校正
//      *                            索引格式：[blockId * TDC_SIZE + tdcIndex]
//      * @param tdcTimeCalTable     TDC时间校正表，用于时间事件的精确校正
//      *                            索引格式：[crystalId * TDC_SIZE + tdcIndex]
//      * @param energyThresholds    能量阈值结构体，包含用于能量计算算法的阈值数组
//      * @param d_energyEventStart  输出：能量事件数组起始指针，存储解析后的能量事件
//      * @param d_timeEventStart    输出：时间事件数组起始指针，存储解析后的时间事件
//      * @param energyCount         输出：解析得到的能量事件总数
//      * @param timeCount           输出：解析得到的时间事件总数
//      *
//      * @return bool 解析是否成功
//      *              - true: 数据包解析成功，所有事件已正确提取
//      *              - false: 解析失败，可能原因包括：
//      *                - 通道号超出有效范围 [MIN_CHANNEL_ID, MAX_CHANNEL_ID]
//      *                - 未知的事件类型标识符(EF)
//      *                - 未知的解析模式
//      *
//      * @note
//      * - 该函数设计为在GPU设备端运行，支持__device__调用
//      * - 函数假定所有输入指针都指向有效的内存区域
//      * - 数据包大小固定为 UDP_RAWDATA_SIZE 字节
//      * - EF=5 标识时间事件，EF=6 标识能量事件
//      * - 完整模式下会进行TDC非线性校正和复杂能量计算
//      *
//      * @warning
//      * - 调用前确保所有指针参数有效且指向足够大的内存区域
//      * - 模板参数MODE必须为有效的resloveMode枚举值
//      * - 输出数组必须有足够空间存储解析的事件
//      *
//      * @see resloveMode 解析模式枚举定义
//      * @see CalcEnergy50100_device() 能量计算算法
//      * @see realChannelToCrystalId(), realChannelToBlockId() 通道映射函数
//      */
//     template <ResloveMode MODE>
//     __device__ bool resolveOneBDM50100Package_device(const char *d_buffer,
//                                                      const PacketPositionInfo *posPtr,
//                                                      float *tdcEnergyCalTable,
//                                                      float *tdcTimeCalTable,
//                                                      const caliCoef::EnergyThresholds_t energyThresholds,
//                                                      caliCoef::EnergyEvent_t *d_energyEventStart,
//                                                      caliCoef::TimeEvent_t *d_timeEventStart,
//                                                      uint32_t &energyCount,
//                                                      uint32_t &timeCount)
//     {
//         bdm50100::DataFrame50100_t *frame = reinterpret_cast<bdm50100::DataFrame50100_t *>(const_cast<char
//         *>(d_buffer)); uint32_t energyId = 0; uint32_t timeId = 0; uint32_t offset = 0; // 偏移量，单位字节个数

//         while (offset < bdm50100::UDP_RAWDATA_SIZE)
//         {
//             uint8_t EFCH1 = frame->data[offset];
//             uint8_t EFCH2 = frame->data[offset + 1];
//             uint16_t channel = ((EFCH1 & 0xF) << 8) + EFCH2;
//             uint8_t EF = (EFCH1 >> 4) & 0xF;

//             // BDM50100的理论local channel上下限， 取决于探测器内部实现
//             if (channel > bdm50100::MAX_CHANNEL_ID || channel < bdm50100::MIN_CHANNEL_ID)
//             {
//                 return false;
//             }

//             if (EF == 5) // 时间事件
//             {
//                 if (MODE == ResloveMode::countOnly)
//                 {
//                     // 不做处理，只计数
//                     timeId++;
//                     offset += sizeof(TimePacket50100_t);
//                 }
//                 else if (MODE == ResloveMode::isOnlyForFineTime)
//                 {
//                     // // 时间包解析与数据准备
//                     // TimePacket50100_t timePacket = raw_frame_copy::copyTimePacket(frame, offset);
//                     // caliCoef::TimeEvent_t temp;

//                     // // 时间事件处理
//                     // temp.bdmId = posPtr->channel;
//                     // temp.channelId = channel;
//                     // temp.T0 = static_cast<double>(timePacket.T0B7);
//                     // temp.deltaT = 0.0f;

//                     // // 将解析后的时间事件存储到输出数组
//                     // d_timeEventStart[timeId] = temp;
//                     // offset += sizeof(TimePacket50100_t);
//                     // timeId++;
//                 }
//                 else if (MODE == ResloveMode::complete)
//                 {
//                     // 时间包解析与数据准备
//                     TimePacket50100_t timePacket = raw_frame_copy::copyTimePacket(frame, offset);
//                     caliCoef::TimeEvent_t temp;

//                     // 时间事件处理
//                     using namespace constant_number;
//                     temp.bdmId = posPtr->channel;
//                     temp.channelId = channel;
//                     double THB = static_cast<double>((timePacket.T0B2 * d2_32 +
//                                                       timePacket.T0B3 * d2_24 +
//                                                       timePacket.T0B4 * d2_16 +
//                                                       timePacket.T0B5 * d2_8 +
//                                                       timePacket.T0B6 * 1.0) *
//                                                      5.0);

//                     const uint32_t tableIndex = index_process::realChannelToCrystalId(channel);
//                     uint8_t realT0B7 =
//                         (timePacket.T0B7 > caliCoef::TDC_SIZE) ? caliCoef::TDC_SIZE : timePacket.T0B7;
//                     float tdcInl = tdcTimeCalTable[tableIndex * caliCoef::TDC_SIZE + realT0B7 - 1];

//                     temp.T0 = THB - static_cast<double>((tdcInl + realT0B7) * 5.0) / caliCoef::K_MAX_BIN;
//                     temp.deltaT = static_cast<float>(timePacket.T2H8 * 5 +
//                                                      static_cast<float>(realT0B7 * 5) / caliCoef::K_MAX_BIN -
//                                                      static_cast<float>(timePacket.T2L8 * 5) / caliCoef::K_MAX_BIN);

//                     // 将解析后的时间事件存储到输出数组
//                     d_timeEventStart[timeId] = temp;
//                     offset += sizeof(TimePacket50100_t);
//                     timeId++;
//                 }
//                 else
//                 {
//                     return false; // 未知模式，直接返回
//                 }
//             }
//             else if (EF == 6)
//             {

//                 if (MODE == ResloveMode::countOnly)
//                 {
//                     // 不做处理，只计数
//                     energyId++;
//                     offset += sizeof(EnergyPacket50100_t);
//                 }
//                 else if (MODE == ResloveMode::isOnlyForFineTime)
//                 {
//                     // // const uint16_t bdmId;
//                     // // 能量包解析与数据准备
//                     // EnergyPacket50100_t energyPacket = raw_frame_copy::copyEnergyPacket(frame, offset);
//                     // caliCoef::EnergyEvent_t temp;

//                     // // 能量事件处理
//                     // temp.bdmId = posPtr->channel;
//                     // temp.channelId = channel;
//                     // temp.T0 = static_cast<double>(energyPacket.T0B7);
//                     // temp.Energy = 0.0f; // 初始值为0

//                     // // 将解析后的时间事件存储到输出数组
//                     // d_energyEventStart[energyId] = temp;
//                     // offset += sizeof(EnergyPacket50100_t);
//                     // energyId++;
//                 }
//                 else if (MODE == ResloveMode::complete)
//                 {
//                     // 能量包解析与数据准备
//                     EnergyPacket50100_t energyPacket = raw_frame_copy::copyEnergyPacket(frame, offset);
//                     caliCoef::EnergyEvent_t temp;
//                     // 能量事件处理
//                     using namespace constant_number;
//                     temp.bdmId = posPtr->channel;
//                     temp.channelId = channel;
//                     double KK_res_bits23 = static_cast<double>((energyPacket.KK1 * d2_16) +
//                                                                (energyPacket.KK2 * d2_8) +
//                                                                energyPacket.KK3);
//                     double QQ_res_bits24 = static_cast<double>((energyPacket.QQ1 * d2_16) +
//                                                                (energyPacket.QQ2 * d2_8) +
//                                                                energyPacket.QQ3);
//                     uint8_t len = static_cast<uint8_t>(energyPacket.RR1 / d2_3);
//                     double RR_res_bits42 = static_cast<double>((energyPacket.RR1 & 3) * d2_40 +
//                                                                energyPacket.RR2 * d2_32 +
//                                                                energyPacket.RR3 * d2_24 +
//                                                                energyPacket.RR4 * d2_16 +
//                                                                energyPacket.RR5 * d2_8 +
//                                                                energyPacket.RR6);
//                     const uint32_t tableIndex = index_process::realChannelToBlockId(channel);
//                     uint8_t realT0B7 = (energyPacket.T0B7 > caliCoef::TDC_SIZE) ? caliCoef::TDC_SIZE :
//                     energyPacket.T0B7; float tdcInl = tdcEnergyCalTable[tableIndex * caliCoef::TDC_SIZE + realT0B7 -
//                     1]; double THB = static_cast<double>((energyPacket.T0B2 * d2_32 +
//                                                       energyPacket.T0B3 * d2_24 +
//                                                       energyPacket.T0B4 * d2_16 +
//                                                       energyPacket.T0B5 * d2_8 +
//                                                       energyPacket.T0B6) *
//                                                      5.0);
//                     temp.T0 = THB - static_cast<double>((tdcInl + realT0B7) * 5.0) / caliCoef::K_MAX_BIN;
//                     temp.Energy = energy_algo::CalcEnergy50100_device(len,
//                                                                       KK_res_bits23,
//                                                                       QQ_res_bits24,
//                                                                       RR_res_bits42,
//                                                                       energyThresholds);

//                     // 将解析后的能量事件存储到输出数组
//                     d_energyEventStart[energyId] = temp;
//                     offset += sizeof(EnergyPacket50100_t);
//                     energyId++;
//                 }
//                 else
//                 {
//                     return false; // 未知模式，直接返回
//                 }
//             }
//             else
//             {
//                 return false; // 未知事件类型，直接返回
//             }
//         }
//         energyCount = energyId;
//         timeCount = timeId;

//         return true; // 解析成功
//     }

//     // 采集模式
//     enum class EventMatchMode : int
//     {
//         NormalCapture = 0,
//         EnergyCalibration,
//         TimeCalibration_Step_1,
//         TimeCalibration_Step_2,
//         TimeCalibration_Step_3,
//         SaturationCalibration,
//         SaturationEmpty,
//         EnergyCalibrationKB2,
//         CrystalTimeResolution,
//         Singles,
//         EnergyCalibrationLYSO,
//         EnergyCalibrationKB3,
//         SaturationCalibration2,
//         SaturationCalibration3
//     };

//     // 一个特定block内的事件匹配函数
//     // 该device函数从energyEventStart、timeEventStart处开始向后搜索[0, energyCount - 1]的能量事件
//     // 搜索范围不超过[0, timeCount - 1]的时间事件
//     // 匹配成功后，填充singleStart[energyId]，并返回

//     __device__ bool eventMatch_device(caliCoef::EnergyEvent_t *energyEventStart,
//                                       caliCoef::TimeEvent_t *timeEventStart,
//                                       const uint32_t energyCount,
//                                       const uint32_t timeCount,
//                                       const bool allowXtalk,
//                                       const float tW,
//                                       const float tShift,
//                                       const caliCoef::EnergyCoef_t *energyTable,
//                                       const caliCoef::TimeCoef_t *timeTable,
//                                       caliCoef::RawGlobalSingle50100_CUDA_t *singleStart)
//     {
//         //=============当前的妥协实现，后续转为模板函数================
//         constexpr EventMatchMode MODE_EM = EventMatchMode::Singles;
//         //========================================================

//         uint32_t timeIdStart = 0; // 时间事件起始索引
//         for (uint32_t energyId = 0; energyId < energyCount; energyId++)
//         {
//             const caliCoef::EnergyEvent_t energyEvent = energyEventStart[energyId];
//             uint32_t matchCount = 0;
//             float totTemp = 0.0f; // 存储最大TOT
//             // uint32_t bestTimeId = 0; // 最佳时间事件索引
//             for (uint32_t timeId = timeIdStart; timeId < timeCount; timeId++)
//             {
//                 const caliCoef::TimeEvent_t timeEvent = timeEventStart[timeId];
//                 float deltaT = energyEvent.T0 - timeEvent.T0 - tShift;
//                 if (deltaT > tW)
//                 {
//                     continue; // 超出时间窗口，跳过
//                 }
//                 else if (deltaT <= tW && deltaT > -1.0 * tW)
//                 {
//                     const float tot = timeEventStart[timeId].deltaT;
//                     if (totTemp < tot)
//                     {
//                         totTemp = tot;        // 更新最大TOT
//                         timeIdStart = timeId; // 更新时间事件起始索引
//                     }
//                     matchCount++;
//                 }
//                 else
//                 {
//                     if (matchCount == 0)
//                     {
//                         timeIdStart = timeId; // 更新时间事件起始索引
//                     }
//                     break; // 时间事件已排序，后续时间事件不可能匹配
//                 }
//             } // 单事件匹配过程

//             if (MODE_EM == EventMatchMode::EnergyCalibration ||
//                 MODE_EM == EventMatchMode::EnergyCalibrationKB2 ||
//                 MODE_EM == EventMatchMode::EnergyCalibrationKB3 ||
//                 MODE_EM == EventMatchMode::SaturationCalibration2 ||
//                 MODE_EM == EventMatchMode::SaturationCalibration3)

//             {
//                 // if (matchCount > 0)
//                 // {
//                 //     // 匹配成功，填充单事件状态
//                 //     caliCoef::RawGlobalSingle50100_CUDA_t single;

//                 //     // caliCoef::SingleStatus_t status = caliCoef::SingleStatus_t({true,
//                 //     // static_cast<uint8_t>(matchCount),
//                 //     //                                                             matchCount > 1,
//                 //     //                                                             0});
//                 //     single.valid = true;
//                 //     single.matchCount = static_cast<uint8_t>(matchCount);
//                 //     single.isXtalk = (matchCount > 1);
//                 //     single.xtalkCount = 0; // 交叉计数初始化为0

//                 //     const caliCoef::TimeEvent_t &timeEvent = timeEventStart[timeIdStart];
//                 //     // single.globalCrystalIndex = globalCryIdOp(timeEvent.bdmId, timeEvent.channelId);
//                 //     const uint16_t realChannelId = index_process::realChannelToCrystalId(timeEvent.channelId);

//                 //     single.timeValue_nano = static_cast<uint64_t>(timeEvent.T0);
//                 //     float energy = energyEvent.Energy;

//                 //     // Energy calibration logic based on mode and match count
//                 //     if (MODE_EM == EventMatchMode::EnergyCalibration && matchCount == 1)
//                 //     {
//                 //         single.energy = energy;
//                 //     }
//                 //     else if (MODE_EM == EventMatchMode::EnergyCalibrationKB2 && matchCount == 2)
//                 //     {
//                 //         energy = energy * energyTable[realChannelId].K;
//                 //         single.energy = energy;
//                 //     }
//                 //     else if (MODE_EM == EventMatchMode::EnergyCalibrationKB3 && matchCount == 3)
//                 //     {
//                 //         energy = energy * energyTable[realChannelId].K;
//                 //         single.energy = energy;
//                 //     }
//                 //     else if ((MODE_EM == EventMatchMode::SaturationCalibration2 ||
//                 //               MODE_EM == EventMatchMode::SaturationCalibration3) &&
//                 //              matchCount > 1)
//                 //     {
//                 //         float KB = 1.0f;
//                 //         if (matchCount == 2)
//                 //         {
//                 //             KB = energyTable[realChannelId].KB2;
//                 //         }
//                 //         else
//                 //         {
//                 //             KB = energyTable[realChannelId].KB3;
//                 //         }
//                 //         energy = energy * energyTable[realChannelId].K * KB;
//                 //         single.energy = energy;
//                 //     }
//                 //     else
//                 //     {
//                 //         single.energy = -1.0f;
//                 //         single.valid = false;
//                 //         single.isXtalk = false;
//                 //         single.xtalkCount = 0; // 无效匹配，设置为异常值
//                 //     }

//                 //     // Store the single event and status
//                 //     singleStart[energyId] = single;
//                 // }
//                 // else
//                 // {

//                 //     singleStart[energyId] = caliCoef::RawGlobalSingle50100_CUDA_t({
//                 //         0,     // timeValue_nano
//                 //         -1.0f, // energy
//                 //         0,     // bdmId
//                 //         0,     // cryId
//                 //         0,     // valid
//                 //         0,     // matchCount
//                 //         0,     // isXtalk
//                 //         0      // xtalkCount
//                 //     });
//                 // }
//             }
//             else
//             {
//                 const bool isMatchValid = (matchCount == 1 && !allowXtalk) || (matchCount > 0 && allowXtalk);
//                 if (isMatchValid)
//                 {
//                     // 匹配成功，填充单事件状态
//                     caliCoef::RawGlobalSingle50100_CUDA_t single;
//                     single.valid = 1; // 设置为有效
//                     single.matchCount = static_cast<uint8_t>(matchCount);
//                     single.isXtalk = (matchCount > 1);
//                     single.xtalkCount = 0; // 交叉计数初始化为0

//                     // 理论上，timeIdStart是匹配的最佳时间事件索引
//                     const caliCoef::TimeEvent_t timeEvent = timeEventStart[timeIdStart];
//                     const uint16_t bdmId = timeEvent.bdmId;
//                     const uint16_t realCryId = index_process::realChannelToCrystalId(timeEvent.channelId);
//                     single.bdmId = bdmId;
//                     single.cryId = realCryId; // 实际晶体编号
//                     single.timeValue_nano = timeEvent.T0 + timeTable[realCryId].B;
//                     float energy = energyEvent.Energy;
//                     float KB = (matchCount == 2 ? energyTable[realCryId].KB2 : energyTable[realCryId].KB3);

//                     if ((energyTable[realCryId].P1 > 1e-6f || energyTable[realCryId].P1 < -1e-6f) && matchCount == 1)
//                     {
//                         energy = energy * energyTable[realCryId].K;
//                         energy = energy * energy * energyTable[realCryId].P1 +
//                                  energy * energyTable[realCryId].P2 +
//                                  energyTable[realCryId].P3;
//                     }
//                     else if (energyTable[realCryId].A2 > 0.0 && matchCount == 2)
//                     {
//                         energy = energy * energyTable[realCryId].K * KB * energyTable[realCryId].A2 +
//                                  energyTable[realCryId].B2;
//                     }
//                     else if (energyTable[realCryId].A3 > 0.0 && matchCount >= 3)
//                     {
//                         energy = energy * energyTable[realCryId].K * KB * energyTable[realCryId].A3 +
//                                  energyTable[realCryId].B3;
//                     }
//                     else
//                     {
//                         energy = energy * energyTable[realCryId].K * (matchCount == 1 ? 1 : KB);
//                     }

//                     if (timeTable[realCryId].TOTA > 0.0f)
//                     {
//                         // 根据最大tot求出对应的能量值
//                         float energyTemp = energy;
//                         if (matchCount > 1)
//                         {
//                             if (totTemp / timeTable[realCryId].TOTA < 1.0)
//                             {
//                                 energyTemp = std::log(1.0 - totTemp / timeTable[realCryId].TOTA) /
//                                                  (-1.0f * timeTable[realCryId].TOTB) +
//                                              timeTable[realCryId].TOTC;
//                             }
//                         }

//                         if (energyTemp < 200)
//                         {
//                             single.timeValue_nano = static_cast<uint64_t>(timeEvent.T0 +
//                                                                           timeTable[realCryId].B +
//                                                                           timeTable[realCryId].B150);
//                         }
//                         else if (energyTemp < 250)
//                         {
//                             single.timeValue_nano = static_cast<uint64_t>(timeEvent.T0 +
//                                                                           timeTable[realCryId].B +
//                                                                           timeTable[realCryId].B200);
//                         }
//                         else if (energyTemp < 300)
//                         {
//                             single.timeValue_nano = static_cast<uint64_t>(timeEvent.T0 +
//                                                                           timeTable[realCryId].B +
//                                                                           timeTable[realCryId].B250);
//                         }
//                         else if (energyTemp < 350)
//                         {
//                             single.timeValue_nano = static_cast<uint64_t>(timeEvent.T0 +
//                                                                           timeTable[realCryId].B +
//                                                                           timeTable[realCryId].B300);
//                         }
//                         else if (energyTemp < 400)
//                         {
//                             single.timeValue_nano = static_cast<uint64_t>(timeEvent.T0 +
//                                                                           timeTable[realCryId].B +
//                                                                           timeTable[realCryId].B350);
//                         }
//                         else if (energyTemp < 421)
//                         {
//                             single.timeValue_nano = static_cast<uint64_t>(timeEvent.T0 +
//                                                                           timeTable[realCryId].B +
//                                                                           timeTable[realCryId].B400);
//                         }
//                     }
//                     else
//                     {

//                     } // 时间值计算完成
//                     single.energy = energy;         // 设置能量值
//                     singleStart[energyId] = single; // 存储单事件数据
//                 }
//                 else
//                 {

//                     singleStart[energyId] = caliCoef::RawGlobalSingle50100_CUDA_t({
//                         0,     // timeValue_nano
//                         -1.0f, // energy
//                         0,     // bdmId
//                         0,     // cryId
//                         0,     // valid
//                         0,     // matchCount
//                         0,     // isXtalk
//                         0      // xtalkCount
//                     });
//                 }
//             }
//         }
//         return true;
//     }

//     /**
//      * @brief CUDA内核函数：并行统计多个BDM50100数据包中的事件数量
//      *
//      * 该函数在GPU上并行处理来自BDM50100探测器的原始数据包，仅统计每个数据包中
//      * 时间事件和能量事件的数量，不进行实际的数据解析和存储。主要用于预先分配
//      * 内存空间或计算前缀和偏移量。
//      *
//      * @param d_buffer     GPU内存中的原始数据包缓冲区，包含packetNum个连续的数据包
//      * @param d_position   数据包位置信息数组，包含每个数据包的BDM ID等位置信息
//      * @param packetNum    要处理的数据包总数，决定启动的线程数量
//      * @param energyCount  输出：每个数据包中能量事件数量的数组 [packetId]
//      * @param timeCount    输出：每个数据包中时间事件数量的数组 [packetId]
//      *
//      * @note
//      * - 该函数使用 countOnly 模式调用 resolveOneBDM50100Package_device
//      * - 每个CUDA线程处理一个数据包，线程ID对应数据包索引
//      * - 函数只统计事件数量，不进行TDC校正、能量计算等复杂处理
//      * - 适用于两阶段处理流程：第一阶段统计，第二阶段解析
//      * - 函数内部会进行边界检查，超出范围的线程会提前返回
//      *
//      * @warning
//      * - 确保d_buffer有足够空间容纳packetNum * UDP_PACKET_SIZE字节
//      * - 确保输出数组energyCount和timeCount有足够空间存储packetNum个元素
//      * - 解析失败的数据包，对应的计数值将被设置为0
//      *
//      * @see resolveOneBDM50100Package_device<countOnly>() 底层计数函数
//      * @see resolveGroupBDM50100_kernel() 完整解析函数
//      * @see PacketPositionInfo 数据包位置信息结构体
//      *
//      * @example
//      * // 典型使用场景：
//      * // 1. 先调用此函数统计事件数量
//      * // 2. 根据统计结果分配内存和计算前缀和
//      * // 3. 再调用resolveGroupBDM50100_kernel进行实际解析
//      */
//     __global__ void countGroupBDM50100_kernel(const char *d_buffer,
//                                               const PacketPositionInfo *d_position,
//                                               const std::size_t packetNum,
//                                               uint32_t *energyCount,
//                                               uint32_t *timeCount)
//     {
//         int tid = blockDim.x * blockIdx.x + threadIdx.x;
//         if (tid >= packetNum)
//         {
//             return; // 超出范围，直接返回
//         }

//         char *d_packetBuffer = const_cast<char *>(d_buffer + tid * bdm50100::UDP_PACKET_SIZE);
//         PacketPositionInfo *posPtr = const_cast<PacketPositionInfo *>(d_position + tid);
//         uint32_t indexEnergy = 0;
//         uint32_t indexTime = 0;

//         // 对于计数模式，创建一个空的能量阈值结构体（不会被实际使用）
//         caliCoef::EnergyThresholds_t emptyThresholds = {};

//         // 解析单个BDM50100数据包
//         if (!resolveOneBDM50100Package_device<ResloveMode::countOnly>(d_packetBuffer,
//                                                                       posPtr,
//                                                                       nullptr,         // tdcEnergyCalTable
//                                                                       nullptr,         // tdcTimeCalTable
//                                                                       emptyThresholds, // energyThresholds (unused in
//                                                                       countOnly mode) nullptr,         //
//                                                                       d_energyEventStart nullptr,         //
//                                                                       d_timeEventStart indexEnergy, indexTime))
//         {
//             // 解析失败，count数组对应位置设为0，直接返回
//             energyCount[tid] = 0; // 更新能量事件计数
//             timeCount[tid] = 0;   // 更新时间事件计数
//             return;
//         }

//         // 解析成功，更新能量和时间事件计数
//         energyCount[tid] = indexEnergy; // 更新能量事件计数
//         timeCount[tid] = indexTime;     // 更新时间事件计数
//         return;
//     }

//     /**
//      * @brief CUDA内核函数：并行解析多个BDM50100数据包，提取时间和能量事件
//      *
//      * 该函数在GPU上并行处理来自BDM50100探测器的原始数据包，将其解析为
//      * 时间事件和能量事件。每个CUDA线程处理一个数据包，使用前缀和数组
//      * 来确定每个线程输出事件的存储位置。
//      *
//      * @param d_buffer            GPU内存中的原始数据包缓冲区，包含packetNum个连续的数据包
//      * @param d_position          数据包位置信息数组，包含每个数据包的BDM ID等位置信息
//      * @param packetNum           要处理的数据包总数，决定启动的线程数量
//      * @param tdcEnergyCalTable   TDC能量校正表，用于能量事件的精确校正
//      *                            索引格式：[blockId * TDC_SIZE + tdcIndex]
//      * @param tdcTimeCalTable     TDC时间校正表，用于时间事件的精确校正
//      *                            索引格式：[crystalId * TDC_SIZE + tdcIndex]
//      * @param energyThresholds    能量阈值结构体，包含用于能量计算算法的阈值数组
//      * @param energyCountPrefix   能量事件数量的前缀和数组，用于确定能量事件输出位置
//      *                            energyCountPrefix[tid]表示第tid个数据包的能量事件在全局数组中的起始索引
//      * @param d_energyEvent       输出：GPU内存中的能量事件数组，存储解析后的能量事件
//      * @param timeCountPrefix     时间事件数量的前缀和数组，用于确定时间事件输出位置
//      *                            timeCountPrefix[tid]表示第tid个数据包的时间事件在全局数组中的起始索引
//      * @param d_timeEvent         输出：GPU内存中的时间事件数组，存储解析后的时间事件
//      *
//      * @note
//      * - 每个CUDA线程处理一个数据包，线程ID对应数据包索引
//      * - 使用前缀和数组避免线程间写入冲突，确保输出数据的正确存储
//      * - 函数内部会进行边界检查，超出范围的线程会提前返回
//      * - 解析失败不会中断其他线程的执行
//      *
//      * @warning
//      * - 确保d_buffer有足够空间容纳packetNum * UDP_PACKET_SIZE字节
//      * - 确保前缀和数组energyCountPrefix和timeCountPrefix已正确计算
//      * - 确保输出数组d_energyEvent和d_timeEvent有足够空间存储所有解析的事件
//      * - 调用前需要先通过countGroupBDM50100_kernel统计事件数量并计算前缀和
//      *
//      * @see countGroupBDM50100_kernel() 事件计数函数，通常在此函数之前调用
//      * @see resolveOneBDM50100Package_device<complete>() 底层解析函数
//      * @see PacketPositionInfo 数据包位置信息结构体
//      * @see EnergyEvent_t, TimeEvent_t 事件数据结构定义
//      *
//      * @example
//      * // 典型使用流程：
//      * // 1. countGroupBDM50100_kernel() - 统计每个包的事件数量
//      * // 2. 计算前缀和数组 energyCountPrefix, timeCountPrefix
//      * // 3. resolveGroupBDM50100_kernel() - 进行实际解析和数据存储
//      */

//     template <ResloveMode MODE>
//     __global__ void resolveGroupBDM50100_kernel(const char *d_buffer,
//                                                 const PacketPositionInfo *d_position,
//                                                 const std::size_t packetNum,
//                                                 const float *tdcEnergyCalTable,
//                                                 const float *tdcTimeCalTable,
//                                                 const caliCoef::EnergyThresholds_t energyThresholds,
//                                                 uint32_t *energyCountPrefix,
//                                                 caliCoef::EnergyEvent_t *d_energyEvent,
//                                                 uint32_t *timeCountPrefix,
//                                                 caliCoef::TimeEvent_t *d_timeEvent)
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
//         caliCoef::EnergyEvent_t *d_energyEventStart = d_energyEvent + energyCountPrefix[tid];
//         caliCoef::TimeEvent_t *d_timeEventStart = d_timeEvent + timeCountPrefix[tid];
//         uint32_t indexEnergy = 0;
//         uint32_t indexTime = 0;

//         // 解析单个BDM50100数据包
//         algo::resolveOneBDM50100Package_device<MODE>(d_packetBuffer,
//                                                      posPtr,
//                                                      tdcEnergyCalTable,
//                                                      tdcTimeCalTable,
//                                                      energyThresholds,
//                                                      d_energyEventStart,
//                                                      d_timeEventStart,
//                                                      indexEnergy,
//                                                      indexTime);

//         return; // 解析成功，返回
//     }

//     __global__ void maskingValidRawSingle_kernel(caliCoef::RawGlobalSingle50100_CUDA_t *d_rawGlobalSingle,
//                                                  uint32_t *d_energyCountPrefix,
//                                                  const uint32_t rawGlobalSingleNum)
//     {
//         int tid = blockDim.x * blockIdx.x + threadIdx.x;

//         // 检查线程ID是否在有效范围内
//         if (tid >= rawGlobalSingleNum)
//         {
//             return; // 超出范围，直接返回
//         }

//         if (d_rawGlobalSingle[tid].valid == 0)
//         {
//             // 如果无效，设置能量计数前缀为0
//             d_energyCountPrefix[tid] = 0;
//         }
//         else
//         {
//             // 如果有效，设置能量计数前缀为1
//             d_energyCountPrefix[tid] = 1;
//         }

//         return;
//     }

//     // __global__ void selectRawSingleByMask_kernel(caliCoef::RawGlobalSingle50100_CUDA_t *d_rawGlobalSingle,
//     //                                              basic::GlobalSingle_t *d_selectedSingle,
//     //                                              const uint32_t *d_prefixedMask,
//     //                                              const uint32_t rawGlobalSingleNum)
//     // {
//     //     int tid = blockDim.x * blockIdx.x + threadIdx.x;
//     //     // 检查线程ID是否在有效范围内
//     //     if (tid >= rawGlobalSingleNum)
//     //     {
//     //         return; // 超出范围，直接返回
//     //     }

//     //     // 根据前缀掩码选择有效的单事件数据
//     //     if (d_rawGlobalSingle[tid].valid)
//     //     {

//     //         // 将有效的RawGlobalSingle转换为GlobalSingle
//     //         basic::GlobalSingle_t temp;
//     //         temp.globalCrystalIndex = d_rawGlobalSingle[tid].bdmId * bdm50100::CRYSTAL_NUM +
//     //                                   d_rawGlobalSingle[tid].cryId;
//     //         temp.timeValue_pico = static_cast<uint64_t>(d_rawGlobalSingle[tid].timeValue_nano * 1000);
//     //         temp.energy = d_rawGlobalSingle[tid].energy;

//     //         d_selectedSingle[d_prefixedMask[tid]] = temp; // 使用前缀掩码索引存储选中的单事件数据
//     //     }
//     //     else
//     //     {
//     //     }

//     //     return;
//     // }
// }