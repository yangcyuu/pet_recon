// #pragma once
// #include <array>
// #include <memory>
// #include <vector>

// #include "../PnI-Config.hpp"
// #include "../basic/PetDataType.h"
// #include "../basic/Point.hpp"
// #include "../process/Acquisition.hpp"
// #include "Detectors.hpp"

// #ifndef PNI_STANDARD_CONFIG_DISABLE_CUDA
// #include <cuda_runtime.h>
// #endif

// namespace openpni::device {
// namespace bdm50100 {
// // BDM50100数据包信息
// constexpr uint64_t UDP_PACKET_SIZE = 1286;                // 1286字节数据包大小
// constexpr uint64_t UDP_RAWDATA_SIZE = 1280;               // 1280字节原始数据大小（字节个数）
// constexpr uint64_t UDP_IP_SIZE = sizeof(uint16_t);        // 2字节IP地址大小
// constexpr uint64_t UDP_UNUSED_SIZE = 4;                   // 4字节未使用数据大小
// constexpr uint64_t MAX_UDP_PACKET_SIZE = UDP_PACKET_SIZE; // 最大数据包大小
// constexpr uint64_t MIN_UDP_PACKET_SIZE = UDP_PACKET_SIZE; // 最小数据包大小

// #pragma pack(push, 1)
// typedef struct DataFrame50100 {
//   uint8_t unused[UDP_UNUSED_SIZE]; // 4字节未使用
//   uint8_t data[UDP_RAWDATA_SIZE];  // 1280字节原始数据
//   uint16_t srcChannel;             // 源通道, ip表示
// } DataFrame50100_t;

// typedef struct TimePacket50100 {
//   uint8_t EFCH1;
//   uint8_t EFCH2;
//   uint8_t T0B2;
//   uint8_t T0B3;
//   uint8_t T0B4;
//   uint8_t T0B5;
//   uint8_t T0B6;
//   uint8_t T0B7;
//   uint8_t T2H8;
//   uint8_t T2L8;
// } TimePacket50100_t;

// typedef struct EnergyPacket50100 {
//   uint8_t EFCH1;
//   uint8_t EFCH2;
//   uint8_t T0B2;
//   uint8_t T0B3;
//   uint8_t T0B4;
//   uint8_t T0B5;
//   uint8_t T0B6;
//   uint8_t T0B7;
//   uint8_t KK1;
//   uint8_t KK2;
//   uint8_t KK3;
//   uint8_t QQ1;
//   uint8_t QQ2;
//   uint8_t QQ3;
//   uint8_t RR1;
//   uint8_t RR2;
//   uint8_t RR3;
//   uint8_t RR4;
//   uint8_t RR5;
//   uint8_t RR6;
// } EnergyPacket50100_t;
// #pragma pack(pop)

// // BDM50100探测器的几何信息
// constexpr uint64_t CRYSTAL_LINE = 6;
// constexpr uint64_t CRYSTAL_NUM_ONE_BLOCK = CRYSTAL_LINE * CRYSTAL_LINE;
// constexpr uint64_t BLOCK_NUM = 8;
// constexpr uint64_t BLOCK_NUM_U = 4;
// constexpr uint64_t BLOCK_NUM_V = 2;
// constexpr uint64_t CRYSTAL_NUM = BLOCK_NUM * CRYSTAL_NUM_ONE_BLOCK;
// // constexpr float BLOCK_PITCH = 26.5;
// // constexpr float CRYSTAL_SIZE = 1.69;

// // BDM50100数据包信息
// constexpr uint64_t SINGLE_BYTES_PER_PACKET = UDP_PACKET_SIZE - UDP_IP_SIZE - UDP_UNUSED_SIZE;               //
// 每个数据包的有效字节数 constexpr uint64_t MAX_ENERGY_EVENT_NUM_PER_PACKET = SINGLE_BYTES_PER_PACKET /
// sizeof(EnergyPacket50100_t); // 最大能量事件数 constexpr uint64_t MAX_TIME_EVENT_NUM_PER_PACKET =
// SINGLE_BYTES_PER_PACKET / sizeof(TimePacket50100_t);     // 最大时间事件数 constexpr uint64_t
// MAX_SINGLE_NUM_PER_PACKET = MAX_ENERGY_EVENT_NUM_PER_PACKET;                             //
// 最大单个数据包中的单个事件数 constexpr uint64_t MIN_SINGLE_NUM_PER_PACKET = 0; // 最小单个数据包中的单个事件数

// constexpr uint64_t MIN_CHANNEL_ID = 1;                                                  // 最小通道ID
// constexpr uint64_t CHANNEL_NUM_PER_BLOCK = CRYSTAL_NUM_ONE_BLOCK + 1;                   //
// 每个block内的通道数（包含一个虚拟通道） constexpr uint64_t MAX_CHANNEL_ID = CHANNEL_NUM_PER_BLOCK * BLOCK_NUM +
// MIN_CHANNEL_ID; // 最大通道ID， 每个block内含有一个虚拟通道（能量事件） } // namespace bdm50100

// namespace bdm50100 {
// class BDM50100Runtime_impl;
// class BDM50100Calibrater_impl;

// }; // namespace bdm50100

// namespace bdm50100 {
// namespace caliCoef {
// constexpr uint64_t TDC_SIZE = 64;
// constexpr uint8_t K_MAX_BIN = 64;              // 细时间5ns划分比
// constexpr uint32_t ENERGY_THRESHOLD_SIZE = 13; // 能量阈值数组大小

// using TDCArray = std::array<float, TDC_SIZE>; // TDC系数数组，64个元素
// using TDCArray_C = float[TDC_SIZE];           // TDC系数数组，C风格数组

// typedef struct EnergyThresholds {
//   float data[ENERGY_THRESHOLD_SIZE]; // 能量阈值数组，13个元素
// } EnergyThresholds_t;

// typedef struct TDCArray_Struct {
//   float data[TDC_SIZE]; // TDC系数数组，64个元素
// } TDCArray_S;           // TDC系数结构体，单个晶体独特

// #pragma pack(push, 2)
// typedef struct EnergyCoef {
//   uint16_t bdmId = 0;     // BDM ID, 原来用于表示IP地址
//   uint16_t channelId = 0; // 通道ID
//   float K = 1.0;          // K系数
//   float KB2 = 1.0f;       // KB2系数
//   float KB3 = 1.0f;       // KB3系数
//   float P1 = 0.0f;        // P1系数
//   float P2 = 0.0f;        // P2系数
//   float P3 = 0.0f;        // P3系数
//   float A2 = 0.0f;        // A2系数
//   float B2 = 0.0f;        // B2系数
//   float A3 = 0.0f;        // A3系数
//   float B3 = 0.0f;        // B3系数
//   float KXA = 0.0f;       // KXA系数
//   float KXB = 0.0f;       // KXB系数
//   float Kempty = 0.0f;    // Kempt系数
// } EnergyCoef_t;           // 能量系数结构体, 单个晶体独特

// typedef struct TimeCoef {
//   uint16_t bdmId = 0;     // BDM ID, 原来用于表示IP地址
//   uint16_t channelId = 0; // 通道ID
//   float B = 0.0f;         // B系数
//   float B150 = 0.0f;      // B150系数
//   float B200 = 0.0f;      // B200系数
//   float B250 = 0.0f;      // B250系数
//   float B300 = 0.0f;      // B300系数
//   float B350 = 0.0f;      // B350系数
//   float B400 = 0.0f;      // B400系数
//   float TOTA = 0.0f;      // TOTA系数
//   float TOTB = 0.0f;      // TOTB系数
//   float TOTC = 0.0f;      // TOTC系数
// } TimeCoef_t;             // 时间系数结构体, 单个晶体独特
// #pragma pack(pop)

// #pragma pack(push, 1)

// typedef struct EnergyEvent {
//   uint16_t bdmId;
//   uint16_t channelId;
//   float Energy;
//   double T0;
// } EnergyEvent_t;

// typedef struct TimeEvent {
//   uint16_t bdmId;
//   uint16_t channelId;
//   float deltaT;
//   double T0;
// } TimeEvent_t;

// // typedef struct RawGlobalSingle50100
// // {
// //     uint32_t globalCrystalIndex;
// //     double timeValue_pico;
// //     float energy;
// //     bool valid;           // 32位对齐
// //     uint8_t matchCount;   // 匹配计数
// //     bool isXtalk;         // 是否为串扰事件
// //     uint8_t xtalkCount;   // 串扰计数，表示串扰事件计算完成的数量
// // } RawGlobalSingle50100_t; // 单事件状态结构体, 用于标记单事件的状态

// // CUDA优化版本的单事件结构体（优化内存布局和对齐）
// typedef struct alignas(
//     8) RawGlobalSingle50100_CUDA {
//   double timeValue_nano; // 8字节，偏移0 (自然8字节对齐)
//   float energy;          // 4字节，偏移8
//   uint16_t bdmId;        // 2字节，偏移12
//   uint16_t cryId;        // 2字节，偏移14
//   uint8_t valid;         // 1字节，偏移16 (0=false, 1=true)
//   uint8_t matchCount;    // 1字节，偏移17
//   uint8_t isXtalk;       // 1字节，偏移18 (0=false, 1=true)
//   uint8_t xtalkCount;    // 1字节，偏移19
//                          // 自动填充4字节到24字节，8字节对齐
//                          // 总计24字节，8字节对齐，空间利用率83.3% (20/24)
// } RawGlobalSingle50100_CUDA_t;

// // // 紧凑版本（24字节，更高的空间利用率83.3%）
// // typedef struct alignas(8) RawGlobalSingle50100_Compact
// // {
// //     double timeValue_pico;       // 8字节，偏移0 (8字节对齐)
// //     uint32_t globalCrystalIndex; // 4字节，偏移8
// //     float energy;                // 4字节，偏移12
// //     uint8_t valid;               // 1字节，偏移16 (使用uint8_t代替bool)
// //     uint8_t matchCount;          // 1字节，偏移17
// //     uint8_t isXtalk;             // 1字节，偏移18 (使用uint8_t代替bool)
// //     uint8_t xtalkCount;          // 1字节，偏移19
// //     uint32_t padding1;           // 4字节填充，偏移20-23
// //     // 总计24字节，8字节对齐，空间利用率更高
// // } RawGlobalSingle50100_Compact_t;
// #pragma pack(pop)

// } // namespace caliCoef

// struct BDM50100CalibrtionTable {
//   std::unique_ptr<std::array<caliCoef::TDCArray_S, BLOCK_NUM>> tdcEnergyCoefs; // TDC能量系数数组
//   std::unique_ptr<std::array<caliCoef::TDCArray_S, CRYSTAL_NUM>> tdcTimeCoefs; // TDC时间系数数组
//   std::unique_ptr<std::array<caliCoef::EnergyCoef, CRYSTAL_NUM>> energyCoefs;  // 能量系数数组
//   std::unique_ptr<std::array<caliCoef::TimeCoef, CRYSTAL_NUM>> timeCoefs;      // 时间系数数组
// };

// // 静态常量邻接矩阵定义 - BDM50100探测器间的邻接关系
// constexpr int ADJACENCY_MATRIX_SIZE = BLOCK_NUM;
// constexpr int ADJACENCY_MATRIX[ADJACENCY_MATRIX_SIZE][ADJACENCY_MATRIX_SIZE] = {
//     {1, 1, 1, 1, 0, 0, 0, 0}, {1, 1, 1, 1, 0, 0, 0, 0}, {1, 1, 1, 1, 1, 1, 0, 0}, {1, 1, 1, 1, 1, 1, 0, 0},
//     {0, 0, 1, 1, 1, 1, 1, 1}, {0, 0, 1, 1, 1, 1, 1, 1}, {0, 0, 0, 0, 1, 1, 1, 1}, {0, 0, 0, 0, 1, 1, 1, 1}};
// }; // namespace bdm50100

// namespace bdm50100 {
// class BDM50100Runtime final : public DetectorBase {
// public:
//   BDM50100Runtime() noexcept;
//   virtual ~BDM50100Runtime() noexcept;

// public:
//   // 校正表相关
//   virtual void loadCalibration(const std::string filename) override;
//   virtual bool isCalibrationLoaded() const noexcept override;

//   // 探测器相关信息
//   virtual DetectorUnchangable detectorUnchangable() const noexcept override;
//   virtual DetectorChangable &detectorChangable() noexcept override;
//   virtual const DetectorChangable &detectorChangable() const noexcept override;

//   // Raw => Singles (CPU未完成)
//   virtual void r2s_cpu() const noexcept override;

//   // 将原始数据转化为单事件，CUDA版本代码
// #ifndef PNI_STANDARD_CONFIG_DISABLE_CUDA
//   virtual void r2s_cuda(const void *d_raw, const PacketPositionInfo *d_position, uint64_t count, basic::LocalSingle_t
//   *d_out) const noexcept override;
// #endif

// private:
//   std::unique_ptr<bdm50100::BDM50100Runtime_impl> m_impl;
// };
// } // namespace bdm50100

// } // namespace openpni::device
