#pragma once
#ifndef _PNI_STD_DPDK_FUNCTIONS_HPP_
#define _PNI_STD_DPDK_FUNCTIONS_HPP_
#include "include/PnI-Config.hpp"
#if PNI_STANDARD_CONFIG_ENABLE_DPDK
#include <arpa/inet.h>
#include <map>
#include <memory>
#include <rte_ether.h>
#include <rte_mbuf.h>
#include <semaphore>

#include "IPSetting.hpp"
#include "include/process/Acquisition.hpp"
#include "src/common/CycledBuffer.hpp"
namespace openpni::process::dpdk {
inline uint32_t ipString2Int(
    const std::string &ip) {
  unsigned char bufipv4[4] = {0};
  uint32_t result = 0;
  if (inet_pton(AF_INET, ip.c_str(), bufipv4) == 1) {
    result = (uint32_t)bufipv4[0] | (uint32_t)bufipv4[1] << 8 | (uint32_t)bufipv4[2] << 16 | (uint32_t)bufipv4[3] << 24;
  }
  return result;
}
inline uint16_t int16ReverseBit(
    const uint16_t &port) {
  return (port << 8 | port >> 8);
}

#if PNI_STANDARD_DPDK_MBUFS
constexpr uint32_t rte_pktmbuf_pool_size{PNI_STANDARD_DPDK_MBUFS};
#else
constexpr uint32_t rte_pktmbuf_pool_size{1024 * 1024 * 4 - 1};
#endif
constexpr uint32_t rte_malloc_retry_num{6};
constexpr uint32_t rte_rx_burst_size{128};
constexpr uint32_t rte_mbuf_cache_size{250};
constexpr uint32_t rte_rx_ring_size{1024};
constexpr uint32_t rte_tx_ring_size{1024};
constexpr uint16_t dpdk_move_burst{64};
constexpr uint16_t tx_rings = 1;
constexpr uint16_t rte_max_supported_socket_num{4};
static_assert(dpdk_move_burst > 0, "dpdk_move_burst must be greater than 0");
constexpr uint32_t dpdk_local_buffer_bytes{uint32_t(dpdk_move_burst) * 1500};
struct DPDKBuffer {
  std::vector<rte_mbuf *> rte_mbufs;
  int portId;
  int num;
};

struct WriteParam {
  uint8_t *src;
  uint16_t *length;
  IP *ipPair;
  uint32_t *offset;
  uint16_t burst;
};
// uint64_t getDPDKBufferNum();
// uint64_t getDPDKBufferAvailable();
uint16_t getDPDKMoveThreadNum();
using FuncWriteToMemory = std::function<void(WriteParam *)>;
bool seizeDPDK(process::FuncLogger &__f, FuncWriteToMemory &); // 抢占DPDK控制权
void releaseDPDK();
}; // namespace openpni::process::dpdk

#endif // _PNI_STD_DPDK_FUNCTIONS_H_

#endif