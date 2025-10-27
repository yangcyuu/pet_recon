#pragma once
#include <regex>
namespace openpni::misc {
inline uint32_t ipStr2ipInt(
    const std::string &ipStr) noexcept {
  std::regex ipRegex(R"((\d+)\.(\d+)\.(\d+)\.(\d+))");
  std::smatch match;
  if (std::regex_match(ipStr, match, ipRegex) && match.size() == 5) {
    try {
      uint32_t ipInt = 0;
      for (int i = 1; i <= 4; ++i) {
        uint32_t octet = std::stoul(match[i].str());
        if (octet > 255)
          return 0; // Invalid IP
        ipInt = (ipInt << 8) | octet;
      }
      return ipInt;
    } catch (const std::exception &) {
      return 0; // Conversion error
    }
  }
  return 0; // Invalid format
}

template <typename Container, typename Range>
inline Container rangeToContainer(
    Range &&range) {
  return Container(std::ranges::begin(range), std::ranges::end(range));
}

} // namespace openpni::misc
