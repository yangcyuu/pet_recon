#pragma once
#include "include/process/Acquisition.hpp"
#include <unordered_map>
namespace openpni::process {
struct IP {
  uint32_t src_addr; // 源IP地址
  uint16_t src_port; // 源端口号
  uint32_t dst_addr; // 目的IP地址
  uint16_t dst_port; // 目的端口号
  bool operator==(const IP &other) const noexcept {
    return src_addr == other.src_addr && src_port == other.src_port &&
           dst_addr == other.dst_addr && dst_port == other.dst_port;
  }
};

inline auto toIPMapper(
    const std::vector<AcquisitionInfo::ChannelSetting> &__channelSettings) noexcept {
  return [channelSettings = __channelSettings](IP ip, uint8_t *udpDatagram,
                                               uint16_t udpLength) -> uint16_t
  // note: return -1 means "unknown or filtered out"
  {
    auto it = std::find_if(channelSettings.begin(), channelSettings.end(),
                           [&ip](const AcquisitionInfo::ChannelSetting &setting) {
                             if (setting.ipSource == 0 && setting.portSource == 0)
                               return setting.ipDestination == ip.dst_addr &&
                                      setting.portDestination == ip.dst_port;
                             else
                               return setting.ipSource == ip.src_addr &&
                                      setting.portSource == ip.src_port &&
                                      setting.ipDestination == ip.dst_addr &&
                                      setting.portDestination == ip.dst_port;
                           });
    if (it == channelSettings.end())
      return -1; // 未找到匹配的通道设置

    if (it->quickFilter &&
        !it->quickFilter(udpDatagram, udpLength, ip.src_addr, ip.src_port))
      return -1;

    return it->channelIndex; // 返回匹配的通道索引
  };
}
} // namespace openpni::process
