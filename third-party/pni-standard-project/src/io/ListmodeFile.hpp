#pragma once
#include "SingleFile.hpp"
#include "include/io/ListmodeIO.hpp"
namespace openpni::io::listmode {
inline bool isValidCrystalIndexType(CrystalIndexType type) noexcept {
  return single::isValidCrystalIndexType(type);
}
inline bool isValidTimeValue1_2Type(TimeValue1_2Type type) noexcept {
  return type == TimeValue1_2Type::ZERO || type == TimeValue1_2Type::INT8 ||
         type == TimeValue1_2Type::INT16;
}
inline ListmodeFileFlagsInvalid invalidFlags(const ListmodeFileHeader &header) noexcept {
  const auto flags4CrystalIndex1 =
      listmode::isValidCrystalIndexType(CrystalIndexType(header.bytes4CrystalIndex1))
          ? ListmodeFileFlagsInvalid::NONE
          : ListmodeFileFlagsInvalid::CrystalIndexType1Invalid;
  const auto flags4CrystalIndex2 =
      listmode::isValidCrystalIndexType(CrystalIndexType(header.bytes4CrystalIndex2))
          ? ListmodeFileFlagsInvalid::NONE
          : ListmodeFileFlagsInvalid::CrystalIndexType2Invalid;
  const auto flags4TimeValue1_2 =
      listmode::isValidTimeValue1_2Type(TimeValue1_2Type(header.bytes4TimeValue1_2))
          ? ListmodeFileFlagsInvalid::NONE
          : ListmodeFileFlagsInvalid::TimeValue1_2TypeInvalid;
  return static_cast<ListmodeFileFlagsInvalid>(
      static_cast<uint16_t>(flags4CrystalIndex1) |
      static_cast<uint16_t>(flags4CrystalIndex2) |
      static_cast<uint16_t>(flags4TimeValue1_2));
}
} // namespace openpni::io::listmode
