#pragma once
#include <cinttypes>

#include "../PnI-Config.hpp"
namespace openpni::basic {
#pragma pack(push, 1)
typedef struct _ufloat16_t {
  int16_t energy;
} UFloat16_t; // 单事件能量采用float16进行存储，其单位为eV；且小于2.0eV的能量无法被正确存储
#pragma pack(pop)

__PNI_CUDA_MACRO__ inline float flt16_flt32(
    const UFloat16_t &__e) {
  union Temp {
    float f;
    int32_t i;
  } temp;
  temp.i = (int32_t(__e.energy) << 13) | 0x40000000;
  return temp.f;
}
__PNI_CUDA_MACRO__ inline UFloat16_t flt32_flt16(
    const float &__f) {
  union Temp {
    float f;
    int32_t i;
  } temp;
  temp.f = __f;
  return UFloat16_t{int16_t(temp.i >> 13)};
}

#pragma pack(push, 1)
typedef struct GlobalSingle {
  using typeof_globalCrystalIndex = unsigned;
  using typeof_timeValue_pico = uint64_t;
  using typeof_energy = float;

  typeof_globalCrystalIndex globalCrystalIndex;
  typeof_timeValue_pico timeValue_pico;
  typeof_energy energy;
} GlobalSingle_t;

typedef struct RigionalSingle {
  using typeof_bdmIndex = uint16_t;
  using typeof_crystalIndex = uint16_t;
  using typeof_energy = float;
  using typeof_timeValue_pico = uint64_t;

  typeof_bdmIndex bdmIndex;
  typeof_crystalIndex crystalIndex;
  typeof_energy energy;
  typeof_timeValue_pico timeValue_pico;
} RigionalSingle_t;

typedef struct LocalSingle {
  unsigned short crystalIndex;
  uint64_t timevalue_pico;
  float energy;
} LocalSingle_t;

typedef struct CoinListmode {
  using typeof_globalCrystalIndex = unsigned;
  using typeof_timeValue1_2 = int16_t;

  typeof_globalCrystalIndex globalCrystalIndex1;
  typeof_globalCrystalIndex globalCrystalIndex2;
  typeof_timeValue1_2 time1_2pico; // Delta time => time1 - time2
} Listmode_t;
#pragma pack(pop)
} // namespace openpni::basic
