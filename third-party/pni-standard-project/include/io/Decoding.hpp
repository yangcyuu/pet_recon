#pragma once
#include <ranges>

#include "ListmodeIO.hpp"
namespace openpni::io::listmode {
struct DecompressImplInfo {
  uint8_t bytes4CrystalIndex1{0}; // 晶体索引1占用的字节数
  uint8_t bytes4CrystalIndex2{0}; // 晶体索引2占用的字节数
  uint8_t bytes4TimeValue1_2{0};  // 时间值1 - 2占用的字节数
  uint64_t count{0};              // 事件个数
};

__PNI_CUDA_MACRO__ inline void decompress_impl(basic::Listmode_t *__out, const char *__inCrystalIndex1,
                                               const char *__inCrystalIndex2, const char *__inTimeValue1_2,
                                               const DecompressImplInfo info, uint64_t index) noexcept;

inline void decompress(
    basic::Listmode_t *__out, const ListmodeFileHeader &__header, const ListmodeSegmentHeader &__segmentHeader,
    const ListmodeSegmentBytes &__bytes) noexcept {
  DecompressImplInfo info;
  info.bytes4CrystalIndex1 = __header.bytes4CrystalIndex1;
  info.bytes4CrystalIndex2 = __header.bytes4CrystalIndex2;
  info.bytes4TimeValue1_2 = __header.bytes4TimeValue1_2;
  info.count = __segmentHeader.count;

  for (uint64_t i = 0; i < info.count; ++i) {
    decompress_impl(__out, __bytes.crystalIndex1Bytes.get(), __bytes.crystalIndex2Bytes.get(),
                    __bytes.timeValue1_2Bytes.get(), info, i);
  }
}

inline std::unique_ptr<basic::Listmode_t[]> decompress(
    const ListmodeFileHeader &__header, const ListmodeSegmentHeader &__segmentHeader,
    const ListmodeSegmentBytes &__bytes) {
  auto result = std::make_unique<basic::Listmode_t[]>(__segmentHeader.count);
  decompress(result.get(), __header, __segmentHeader, __bytes);
  return result;
}

// #ifndef PNI_STANDARD_CONFIG_DISABLE_CUDA
// #if __CUDACC__
// struct ListmodeSegmentBytes_CUDA {
//   cuda_sync_ptr<char> crystalIndex1Bytes; // 晶体索引1
//   cuda_sync_ptr<char> crystalIndex2Bytes; // 晶体索引2
//   cuda_sync_ptr<char> timeValue1_2Bytes;  // 时间值1 - 2
// };
// inline ListmodeSegmentBytes_CUDA to_cuda_segment_bytes(
//     const ListmodeSegmentBytes &__bytes) {
//   ListmodeSegmentBytes_CUDA result;
//   result.crystalIndex1Bytes = make_cuda_sync_ptr_from_hcopy(
//       std::span<const char>(&__bytes.crystalIndex1Bytes[0], __bytes.storagedBytesCrystalIndex1));
//   result.crystalIndex2Bytes = make_cuda_sync_ptr_from_hcopy(
//       std::span<const char>(&__bytes.crystalIndex2Bytes[0], __bytes.storagedBytesCrystalIndex2));
//   result.timeValue1_2Bytes = make_cuda_sync_ptr_from_hcopy(
//       std::span<const char>(&__bytes.timeValue1_2Bytes[0], __bytes.storagedBytesTimeValue1_2));
//   return result;
// }

// __global__ void kernel_decompress(
//     basic::Listmode_t *__out,
//     const char *__inCrystalIndex1,
//     const char *__inCrystalIndex2,
//     const char *__inTimeValue1_2,
//     const DecompressImplInfo info) noexcept
// {
//     uint64_t index = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
//     decompress_impl(__out, __inCrystalIndex1, __inCrystalIndex2, __inTimeValue1_2,
//     info, index);
// }

// inline void decompress(
//     const basic::cuda_unique_ptr<basic::Listmode_t> &__out,
//     const ListmodeSegmentBytes_CUDA &__bytes,
//     const ListmodeFileHeader &__header,
//     const ListmodeSegmentHeader &__segmentHeader) noexcept
// {
//     DecompressImplInfo info;
//     info.bytes4CrystalIndex1 = __header.bytes4CrystalIndex1;
//     info.bytes4CrystalIndex2 = __header.bytes4CrystalIndex2;
//     info.bytes4TimeValue1_2 = __header.bytes4TimeValue1_2;
//     info.count = __segmentHeader.count;

//     uint64_t gridSize = (info.count + 255) / 256; // 每个线程块256个线程

//     kernel_decompress<<<gridSize, 256>>>(
//         __out.get(),
//         __bytes.crystalIndex1Bytes.get(),
//         __bytes.crystalIndex2Bytes.get(),
//         __bytes.timeValue1_2Bytes.get(),
//         info);
// }

// inline basic::cuda_unique_ptr<basic::Listmode_t> decompress(
//     const ListmodeSegmentBytes_CUDA &__bytes,
//     const ListmodeFileHeader &__header,
//     const ListmodeSegmentHeader &__segmentHeader)
// {
//     basic::cuda_unique_ptr<basic::Listmode_t> result =
//     basic::make_cuda_sync_ptr<basic::Listmode_t>(
//         __segmentHeader.count);
//     decompress(result, __bytes, __header, __segmentHeader);
//     return result;
// }
// #endif
// #endif

// } // namespace openpni::io::listmode

__PNI_CUDA_MACRO__ inline void decompress_impl(
    basic::Listmode_t *__out, const char *__inCrystalIndex1, const char *__inCrystalIndex2,
    const char *__inTimeValue1_2, const DecompressImplInfo info, uint64_t index) noexcept {
  if (index >= info.count)
    return;

  auto out = __out + index;
  switch (CrystalIndexType(info.bytes4CrystalIndex1)) {
  case CrystalIndexType::UINT16:
    out->globalCrystalIndex1 = *reinterpret_cast<const uint16_t *>(__inCrystalIndex1 + index * 2);
    break;
  case CrystalIndexType::UINT24:
    out->globalCrystalIndex1 = *reinterpret_cast<const uint32_t *>(__inCrystalIndex1 + index * 3) & 0x00FFFFFF;
    break;
  case CrystalIndexType::UINT32:
    out->globalCrystalIndex1 = *reinterpret_cast<const uint32_t *>(__inCrystalIndex1 + index * 4);
    break;
  default:
    out->globalCrystalIndex1 = 0; // 无效的晶体索引类型
    break;
  }
  switch (CrystalIndexType(info.bytes4CrystalIndex2)) {
  case CrystalIndexType::UINT16:
    out->globalCrystalIndex2 = *reinterpret_cast<const uint16_t *>(__inCrystalIndex2 + index * 2);
    break;
  case CrystalIndexType::UINT24:
    out->globalCrystalIndex2 = *reinterpret_cast<const uint32_t *>(__inCrystalIndex2 + index * 3) & 0x00FFFFFF;
    break;
  case CrystalIndexType::UINT32:
    out->globalCrystalIndex2 = *reinterpret_cast<const uint32_t *>(__inCrystalIndex2 + index * 4);
    break;
  default:
    out->globalCrystalIndex2 = 0; // 无效的晶体索引类型
    break;
  }
  switch (TimeValue1_2Type(info.bytes4TimeValue1_2)) {
  case TimeValue1_2Type::ZERO:
    out->time1_2pico = 0; // 无时间差
    break;
  case TimeValue1_2Type::INT8:
    out->time1_2pico = static_cast<int16_t>(*reinterpret_cast<const int8_t *>(__inTimeValue1_2 + index * 1)) *
                       256; // 时间差为value * 256皮秒
    break;
  case TimeValue1_2Type::INT16:
    out->time1_2pico = *reinterpret_cast<const int16_t *>(__inTimeValue1_2 + index * 2); // 时间差为value * 1皮秒
    break;
  default:
    out->time1_2pico = 0; // 无效的时间值类型
    break;
  }
}
} // namespace openpni::io::listmode

namespace openpni::io::single {
struct DecompressImplInfo {
  uint8_t bytes4CrystalIndex{0}; // 每个晶体索引占用的字节数
  uint8_t bytes4TimeValue{0};    // 每个时间值占用的字节数
  uint8_t bytes4Energy{0};       // 每个能量值占用的字节数
  uint64_t count{0};             // 数据包数量
};

__PNI_CUDA_MACRO__ inline void decompress_impl(basic::GlobalSingle_t *__out, const char *__inCrysalIndex,
                                               const char *__inTimeValue, const char *__inEnergy,
                                               const DecompressImplInfo info, uint64_t index) noexcept;

inline void decompress(
    basic::GlobalSingle_t *__out, const SingleSegmentBytes &__bytes, const SingleFileHeader &__header,
    const SingleSegmentHeader &__segmentHeader) noexcept {
  DecompressImplInfo info;
  info.bytes4CrystalIndex = __header.bytes4CrystalIndex;
  info.bytes4TimeValue = __header.bytes4TimeValue;
  info.bytes4Energy = __header.bytes4Energy;
  info.count = __segmentHeader.count;
  for (auto index : std::views::iota(0ull, __segmentHeader.count)) {
    decompress_impl(__out, __bytes.crystalIndexBytes.get(), __bytes.timeValueBytes.get(), __bytes.energyBytes.get(),
                    info, index);
  }
}

inline std::unique_ptr<basic::GlobalSingle_t[]> decompress(
    const SingleSegmentBytes &__bytes, const SingleFileHeader &__header, const SingleSegmentHeader &__segmentHeader) {
  std::unique_ptr<basic::GlobalSingle_t[]> result = std::make_unique<basic::GlobalSingle_t[]>(__segmentHeader.count);
  decompress(result.get(), __bytes, __header, __segmentHeader);
  return result;
}

#ifndef PNI_STANDARD_CONFIG_DISABLE_CUDA
#ifdef __CUDACC__
// struct SingleSegmentBytes_CUDA
// {
//     basic::cuda_unique_ptr<char> crystalIndexBytes; // 晶体索引字节数组
//     basic::cuda_unique_ptr<char> timeValueBytes;    // 时间值字节数组
//     basic::cuda_unique_ptr<char> energyBytes;       // 能量字节数组
// };

// inline SingleSegmentBytes_CUDA to_cuda_segment_bytes(
//     const SingleSegmentBytes &bytes) noexcept
// {
//     SingleSegmentBytes_CUDA result;
//     result.crystalIndexBytes = basic::cuda_unique_ptr<char>(
//         bytes.crystalIndexBytes.get(), bytes.storagedBytesCrystalIndex);
//     result.timeValueBytes = basic::cuda_unique_ptr<char>(
//         bytes.timeValueBytes.get(), bytes.storagedBytesTimeValue);
//     result.energyBytes = basic::cuda_unique_ptr<char>(
//         bytes.energyBytes.get(), bytes.storagedBytesEnergy);
//     return result;
// }

// __global__ inline void kernel_decompress(
//     basic::GlobalSingle_t *__out,
//     const char *__inCrysalIndex,
//     const char *__inTimeValue,
//     const char *__inEnergy,
//     const DecompressImplInfo info) noexcept
// {
//     uint64_t index = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
//     decompress_impl(__out, __inCrysalIndex, __inTimeValue, __inEnergy, info, index);
// }

// inline void decompress(
//     const basic::cuda_unique_ptr<basic::GlobalSingle_t> &__out,
//     const SingleSegmentBytes_CUDA &__bytes,
//     const SingleFileHeader &__header,
//     const SingleSegmentHeader &__segmentHeader) noexcept
// {
//     DecompressImplInfo info;
//     info.bytes4CrystalIndex = __header.bytes4CrystalIndex;
//     info.bytes4TimeValue = __header.bytes4TimeValue;
//     info.bytes4Energy = __header.bytes4Energy;
//     info.count = __segmentHeader.count;

//     uint64_t gridSize = (info.count + 255) / 256; // 每个线程块256个线程

//     kernel_decompress<<<gridSize, 256>>>(
//         __out.get(),
//         __bytes.crystalIndexBytes.get(),
//         __bytes.timeValueBytes.get(),
//         __bytes.energyBytes.get(),
//         info);
// }

// inline basic::cuda_unique_ptr<basic::GlobalSingle_t> decompress(
//     const SingleSegmentBytes_CUDA &__bytes,
//     const SingleFileHeader &__header,
//     const SingleSegmentHeader &__segmentHeader)
// {
//     basic::cuda_unique_ptr<basic::GlobalSingle_t> result =
//     basic::make_cuda_sync_ptr<basic::GlobalSingle_t>(
//         __segmentHeader.count);
//     decompress(result, __bytes, __header, __segmentHeader);
//     return result;
// }
#endif // __CUDA_RUNTIME_H__
#endif // !PNI_STANDARD_CONFIG_DISABLE_CUDA

} // namespace openpni::io::single

__PNI_CUDA_MACRO__ inline void openpni::io::single::decompress_impl(
    basic::GlobalSingle_t *__out, const char *__inCrysalIndex, const char *__inTimeValue, const char *__inEnergy,
    const DecompressImplInfo info, uint64_t index) noexcept {
  if (index >= info.count)
    return;

  auto *out = __out + index;
  switch (CrystalIndexType(info.bytes4CrystalIndex)) {
  case CrystalIndexType::UINT16:
    out->globalCrystalIndex = *reinterpret_cast<const uint16_t *>(__inCrysalIndex + index * 2);
    break;
  case CrystalIndexType::UINT24:
    out->globalCrystalIndex = *reinterpret_cast<const uint32_t *>(__inCrysalIndex + index * 3) & 0x00FFFFFF;
    break;
  case CrystalIndexType::UINT32:
    out->globalCrystalIndex = *reinterpret_cast<const uint32_t *>(__inCrysalIndex + index * 4);
    break;
  default:
    out->globalCrystalIndex = 0; // 无效的晶体索引类型
  }

  switch (TimeValueType(info.bytes4TimeValue)) {
  case TimeValueType::UINT32:
    out->timeValue_pico = *reinterpret_cast<const uint32_t *>(__inTimeValue + index * 4);
    break;
  case TimeValueType::UINT40:
    out->timeValue_pico = *reinterpret_cast<const uint64_t *>(__inTimeValue + index * 5) & 0x000000FFFFFFFFFF;
    break;
  case TimeValueType::UINT48:
    out->timeValue_pico = *reinterpret_cast<const uint64_t *>(__inTimeValue + index * 6) & 0x0000FFFFFFFFFFFF;
    break;
  case TimeValueType::UINT56:
    out->timeValue_pico = *reinterpret_cast<const uint64_t *>(__inTimeValue + index * 7) & 0x00FFFFFFFFFFFFFF;
    break;
  case TimeValueType::UINT64:
    out->timeValue_pico = *reinterpret_cast<const uint64_t *>(__inTimeValue + index * 8);
    break;
  default:
    out->timeValue_pico = 0; // 无效的时间值类型
  }

  switch (EnergyType(info.bytes4Energy)) {
  case EnergyType::ZERO:
    out->energy = 511.0e3f; // 能量总是511keV
    break;
  case EnergyType::UINT8:
    out->energy = float(*reinterpret_cast<const uint8_t *>(__inEnergy + index * 1)) * 4.0e3f; // 能量倍率为4keV
    break;
  case EnergyType::UFLT16:
    out->energy = basic::flt16_flt32(*reinterpret_cast<const basic::UFloat16_t *>(__inEnergy + index * 2));
    break;
  case EnergyType::FLT32:
    out->energy = *reinterpret_cast<const float *>(__inEnergy + index * 4);
    break;
  default:
    out->energy = 0.0f; // 无效的能量类型
  }
}
