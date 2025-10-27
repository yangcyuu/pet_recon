#pragma once
#include <memory>

#include "../Exceptions.hpp"
#include "../PnI-Config.hpp"
#include "../basic/PetDataType.h"
#include "../detector/BDM50100Array.hpp"
#if !PNI_STANDARD_CONFIG_DISABLE_CUDA

namespace openpni::device::deviceArray::bdm50100Array::layout {
namespace D930 {
namespace constants {
constexpr int32_t BDM_PER_RING = 48; // 每个环路的BDM数量
constexpr int32_t RING_NUM = 3;      // 环路数量
constexpr int32_t BASE_RAOTATE_MODULE = -1;

constexpr int32_t ZWCrystalsInX_ = 1;   // 每个BDM的Y方向晶体数量
constexpr int32_t ZWCrystalsInY_ = 6;   // 每个BDM的Y方向晶体数量
constexpr int32_t ZWCrystalsInZ_ = 6;   // 每个BDM的Z方向晶体数量
constexpr int32_t ZWSubmodulesInY_ = 2; // 每个BDM的Y方向子模块数量
constexpr int32_t ZWSubmodulesInZ_ = 4; // 每个BDM的Y方向子模块数量
constexpr int32_t ZWModulesInY_ = 1;
constexpr int32_t ZWModulesInZ_ = 3;                                                     // 每个BDM的Y方向模块数量
constexpr int32_t ZWCrystalsPerRing_ = BDM_PER_RING * ZWCrystalsInY_ * ZWSubmodulesInY_; // 每个环路的晶体数量
constexpr int32_t ZWOffset_ = 0;
} // namespace constants

__host__ __device__ inline uint32_t getGlobalCryId(
    const uint16_t &bdmId, const uint16_t &cryId) {
  using namespace constants;
  int32_t reference[36] = {25, 30, 18, 7,  12, 0, 27, 32, 20, 9,  14, 2, 29, 34, 22, 11, 16, 4,
                           31, 19, 24, 13, 1,  6, 33, 21, 26, 15, 3,  8, 35, 23, 28, 17, 5,  10};
  int32_t channel[36];
  for (int32_t i = 0; i < 36; ++i) {
    channel[reference[i]] = i;
  }

  int32_t ring1 = 0;
  int32_t crystal1 = 0;
  int32_t crystalID1 = 0;
  int32_t submoduleID1 = 0;
  int32_t moduleID1 = 0;
  int32_t rsectorID1 = 0;

  int32_t cryIdInBlock = cryId % CRYSTAL_NUM_ONE_BLOCK; // 确保cryId在0-35范围内
  crystalID1 = channel[cryIdInBlock];
  submoduleID1 = cryId / CRYSTAL_NUM_ONE_BLOCK;
  moduleID1 = RING_NUM - 1 - bdmId / BDM_PER_RING;
  rsectorID1 = (BASE_RAOTATE_MODULE + bdmId % BDM_PER_RING + 1 + BDM_PER_RING - 1) % BDM_PER_RING;

  int32_t submoduleIDTemp1 = 0;
  if (submoduleID1 == 0 || submoduleID1 == 3 || submoduleID1 == 4 || submoduleID1 == 7) {
    submoduleIDTemp1 = 1;
  }

  // 从AXIS1 晶体最后一排开始排ring
  ring1 = (5 - crystalID1 / ZWCrystalsInY_) + (submoduleID1 / ZWSubmodulesInY_) * ZWCrystalsInZ_ +
          (moduleID1 / ZWModulesInY_) * ZWSubmodulesInZ_ * ZWCrystalsInZ_;

  // 单环晶体条编号 从后往前
  crystal1 = (rsectorID1 * ZWCrystalsInY_ * ZWModulesInY_ * ZWSubmodulesInY_) +
             ((submoduleIDTemp1 % ZWSubmodulesInY_) * ZWCrystalsInY_) + (5 - crystalID1 % ZWCrystalsInY_);

  crystal1 = (-crystal1 - 1 + ZWOffset_ + ZWCrystalsPerRing_) % ZWCrystalsPerRing_ + ring1 * ZWCrystalsPerRing_;

  return crystal1;
}
} // namespace D930
} // namespace openpni::device::deviceArray::bdm50100Array::layout

#endif