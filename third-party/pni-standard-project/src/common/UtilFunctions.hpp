#pragma once
#include <filesystem>

namespace openpni::common {
inline std::filesystem::space_info
getSpaceInfo_noexcept(const std::string &path) noexcept {
  try {
    return std::filesystem::space(path);
  } catch (const std::filesystem::filesystem_error &e) {
    return {}; // Return zero space info on error
  }
}
} // namespace openpni::common
