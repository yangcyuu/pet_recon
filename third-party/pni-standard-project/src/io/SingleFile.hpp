#pragma once
#include "include/io/IO.hpp"
#include "src/common/TemplateFunctions.hpp"

namespace openpni::io::single {
inline bool isValidCrystalIndexType(
    CrystalIndexType type) noexcept {
  return common::is_one_of(type, {CrystalIndexType::UINT16, CrystalIndexType::UINT24, CrystalIndexType::UINT32});
}
inline bool isValidTimeValueType(
    TimeValueType type) noexcept {
  return common::is_one_of(type, {TimeValueType::UINT32, TimeValueType::UINT40, TimeValueType::UINT48,
                                  TimeValueType::UINT56, TimeValueType::UINT64});
}
inline bool isValidEnergyType(
    EnergyType type) noexcept {
  return common::is_one_of(type, {EnergyType::ZERO, EnergyType::UINT8, EnergyType::UFLT16, EnergyType::FLT32});
}
inline SingleFileFlagsInvalid invalidFlags(
    const SingleFileHeader &header) noexcept {
  const auto bits4CrystalIndex = isValidCrystalIndexType(CrystalIndexType(header.bytes4CrystalIndex))
                                     ? SingleFileFlagsInvalid::NONE
                                     : SingleFileFlagsInvalid::CRYSTAL_INDEX;
  const auto bits4TimeValue = isValidTimeValueType(TimeValueType(header.bytes4TimeValue))
                                  ? SingleFileFlagsInvalid::NONE
                                  : SingleFileFlagsInvalid::TIME_VALUE;
  const auto bits4Energy = isValidEnergyType(EnergyType(header.bytes4Energy)) ? SingleFileFlagsInvalid::NONE
                                                                              : SingleFileFlagsInvalid::ENERGY;
  return static_cast<SingleFileFlagsInvalid>(static_cast<uint8_t>(bits4CrystalIndex) |
                                             static_cast<uint8_t>(bits4TimeValue) | static_cast<uint8_t>(bits4Energy));
}
} // namespace openpni::io::single
