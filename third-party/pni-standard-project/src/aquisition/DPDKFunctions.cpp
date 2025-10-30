#include "include/PnI-Config.hpp"
#if PNI_STANDARD_CONFIG_ENABLE_DPDK
#include <chrono>
#include <iostream>
#include <rte_arp.h>
#include <rte_common.h>
#include <rte_cycles.h>
#include <rte_eal.h>
#include <rte_ethdev.h>
#include <rte_launch.h>
#include <rte_lcore.h>
#include <rte_per_lcore.h>
#include <set>
#include <shared_mutex>
#include <thread>

#include "DPDKFunctions.hpp"
#include "include/process/Acquisition.hpp"
#include "src/common/AtomicBuffer.hpp"
struct PortInfo {
  rte_ether_addr mac;
};
uint32_t reverseBytes(
    uint32_t a) {
  uint8_t a4 = a >> 24, a3 = a >> 16 & 0xff, a2 = a >> 8 & 0xff, a1 = a & 0xff;
  return a1 << 24 | a2 << 16 | a3 << 8 | a4;
}
namespace dpdk = openpni::process::dpdk;
enum FunctionType { Rx, Move, Copy };
struct MoveBuffer {
  std::array<rte_mbuf *, dpdk::rte_rx_burst_size> rte_mbufs;
  int num;
};
struct LCoreInfo {
  FunctionType lcore_work_type;
  int port_id;
  uint32_t ipv4;
  unsigned ring_id;
  std::unique_ptr<openpni::common::AtomicRingBuffer<MoveBuffer>> move_buffer;
};

static std::map<unsigned, LCoreInfo> dpdk_coreInfo;
// static rte_mempool *dpdk_mempool;
static PortInfo dpdk_port_info[dpdk::rte_max_supported_port_num];
static std::unique_ptr<openpni::common::NoDeadWaitCycledBuffer<dpdk::DPDKBuffer>> dpdk_ring_buffer;
static std::shared_mutex dpdk_seizeMutex;
static bool dpdk_stopFlag = false;
static dpdk::DPDKParam dpdk_param;
static int total_rx_thread_num = 0;

dpdk::FuncWriteToMemory *dpdk_writeToMemory;

uint8_t openpni::process::dpdk::rte_port_num;
openpni::process::dpdk::PortStatus openpni::process::dpdk::dpdk_status[dpdk::rte_max_supported_port_num];
rte_mempool *dpdk_mempool[dpdk::rte_max_supported_port_num];
std::atomic<uint64_t> dpdk_buffer_free[dpdk::rte_max_supported_port_num]{0};
uint64_t dpdk_buffer_total[dpdk::rte_max_supported_port_num]{0};

void encode_arp_pkt(
    uint8_t *msg, uint8_t *dst_mac, uint32_t sip, uint32_t tip, uint16_t port_id, rte_ether_addr *src_mac) {
  struct rte_ether_hdr *ether_hdr = (struct rte_ether_hdr *)msg;

  rte_memcpy(ether_hdr->dst_addr.addr_bytes, dst_mac, RTE_ETHER_ADDR_LEN);
  rte_memcpy(ether_hdr->src_addr.addr_bytes, src_mac->addr_bytes, RTE_ETHER_ADDR_LEN);
  ether_hdr->ether_type = htons(RTE_ETHER_TYPE_ARP);

  struct rte_arp_hdr *arp_hdr = (struct rte_arp_hdr *)(ether_hdr + 1);
  arp_hdr->arp_protocol = htons(RTE_ETHER_TYPE_IPV4);
  arp_hdr->arp_plen = sizeof(uint32_t);
  arp_hdr->arp_opcode = htons(2);
  arp_hdr->arp_hardware = htons(1);
  arp_hdr->arp_hlen = RTE_ETHER_ADDR_LEN;

  rte_memcpy(arp_hdr->arp_data.arp_sha.addr_bytes, src_mac->addr_bytes, RTE_ETHER_ADDR_LEN);
  rte_memcpy(arp_hdr->arp_data.arp_tha.addr_bytes, dst_mac, RTE_ETHER_ADDR_LEN);

  arp_hdr->arp_data.arp_tip = tip;
  arp_hdr->arp_data.arp_sip = sip;
}
struct rte_mbuf *create_arp_packet(
    struct rte_mempool *mbuf_pool, uint8_t *dst_mac, uint32_t sip, uint32_t dip, uint16_t port_id,
    rte_ether_addr *src_mac) {
  const uint32_t total_length = sizeof(struct rte_ether_hdr) + sizeof(struct rte_arp_hdr);

  struct rte_mbuf *mbuf = rte_pktmbuf_alloc(mbuf_pool);
  if (mbuf) {
    mbuf->pkt_len = total_length;
    mbuf->data_len = total_length;
    uint8_t *pkt_data = rte_pktmbuf_mtod(mbuf, uint8_t *);
    encode_arp_pkt(pkt_data, dst_mac, sip, dip, port_id, src_mac);
  }
  return mbuf;
}

int dpdk_rx(
    void *UNUSED__) {
  /* Main work of application loop. 8< */
  using dpdk::rte_rx_burst_size;

  thread_local auto socket_id = rte_socket_id();
  thread_local const auto lcore_id = rte_lcore_id();
  std::cerr << "DPDK RX thread started on lcore " + std::to_string(lcore_id) + " on socket " +
                   std::to_string(socket_id) + "\n";
  auto &lcore_info = dpdk_coreInfo[lcore_id];
  auto &portStatus = openpni::process::dpdk::dpdk_status[lcore_info.port_id];
  const int portId = lcore_info.port_id;

  using namespace std::chrono;
  auto clockNow = steady_clock::now();
  constexpr uint32_t timeResetArp = 1000;
  thread_local bool arpFirstSent = false;
  thread_local const uint16_t ring_id = lcore_info.ring_id;

  lcore_info.move_buffer = std::make_unique<openpni::common::AtomicRingBuffer<MoveBuffer>>(
      dpdk::rte_pktmbuf_pool_size / dpdk::rte_rx_burst_size / total_rx_thread_num);

  uint64_t key = 0;

  auto handle_pkt_mbuf = [&lcore_info, portId](rte_mbuf *mbuf) -> rte_mbuf * {
    rte_ether_hdr *ethHeader{rte_pktmbuf_mtod(mbuf, struct rte_ether_hdr *)};

    if (!ring_id && ethHeader->ether_type == rte_cpu_to_be_16(RTE_ETHER_TYPE_ARP)) {
      const auto ipv4 = lcore_info.ipv4;
      rte_arp_hdr *arp_hdr = rte_pktmbuf_mtod_offset(mbuf, struct rte_arp_hdr *, sizeof(struct rte_ether_hdr));
      in_addr targetAddr, sourceAddr;
      targetAddr.s_addr = arp_hdr->arp_data.arp_tip;
      sourceAddr.s_addr = arp_hdr->arp_data.arp_sip;

      if (ipv4 == targetAddr.s_addr) {
        // Target IP is my ip.
        rte_mbuf *arp_buf =
            create_arp_packet(dpdk_mempool[portId], arp_hdr->arp_data.arp_sha.addr_bytes, ipv4,
                              arp_hdr->arp_data.arp_sip, portId, (struct rte_ether_addr *)&dpdk_port_info[portId].mac);
        if (arp_buf) {
          rte_eth_tx_burst(portId, 0, &arp_buf, 1);
          rte_pktmbuf_free(arp_buf);
        }
      }
      rte_pktmbuf_free(mbuf);
      dpdk_buffer_free[portId]++;
      return nullptr;
    } else if (ethHeader->ether_type == rte_cpu_to_be_16(RTE_ETHER_TYPE_IPV4)) {
      return mbuf;
    } else {
      rte_pktmbuf_free(mbuf);
      dpdk_buffer_free[portId]++;
      return nullptr;
    }
    return nullptr;
  };

  bool lastLoopZero = false;
  while (!dpdk_stopFlag) {
    if (!ring_id) // No need to send ARP in ring != 0.
      if (const auto _clockNow = steady_clock::now();
          duration_cast<milliseconds>(_clockNow - clockNow).count() > timeResetArp || !arpFirstSent) {
        arpFirstSent = true;
        clockNow = _clockNow;
        /** Arp announcement!!!
         * 当设备需要使用一个新的IP地址时，它会发送一个ARP通告包。这个包的发送者IP和目标IP都设置为设备自身的IP地址，目标MAC地址设置为全零。这样，局域网中的所有设备都会收到这个ARP请求，并更新其ARP缓存
         */
        struct rte_ether_addr allZeroDst;
        for (int _i = 0; _i < RTE_ETHER_ADDR_LEN; _i++)
          allZeroDst.addr_bytes[_i] = 0x00;
        struct rte_mbuf *arp_buf =
            create_arp_packet(dpdk_mempool[portId], allZeroDst.addr_bytes, lcore_info.ipv4, lcore_info.ipv4, portId,
                              (struct rte_ether_addr *)&dpdk_port_info[portId].mac);
        if (arp_buf) {
          rte_eth_tx_burst(portId, 0, &arp_buf, 1);
          rte_pktmbuf_free(arp_buf);
        }

        rte_eth_stats_get(portId, &portStatus.rte_stats);
        portStatus.dpdkBufferVolume = dpdk_buffer_total[portId];
        portStatus.dpdkBufferFree = dpdk_buffer_free[portId];
      }

    rte_mbuf *temp_mbufs[rte_rx_burst_size];
    rte_mbuf *udp_mbufs[rte_rx_burst_size];
    const uint16_t actual_rx = rte_eth_rx_burst(portId, ring_id, temp_mbufs, rte_rx_burst_size);

    if (actual_rx == 0) {
      if (lastLoopZero)
        continue;
      lastLoopZero = true;
    }

    dpdk_buffer_free[portId] -= actual_rx;
    int udp_mbuf_num = 0;
    for (int i = 0; i < actual_rx; i++) {
      rte_mbuf *mbuf = handle_pkt_mbuf(temp_mbufs[i]);
      if (mbuf)
        udp_mbufs[udp_mbuf_num++] = mbuf;
    }

    lcore_info.move_buffer->write([&](MoveBuffer &buffer) noexcept {
      buffer.num = udp_mbuf_num;
      for (int i = 0; i < udp_mbuf_num; i++) {
        buffer.rte_mbufs[i] = udp_mbufs[i];
      }
    });
  }
  lcore_info.move_buffer->stop();
  std::cerr << "DPDK RX thread on lcore " + std::to_string(lcore_id) + " exited.\n";
  return 0;
}
int dpdk_move(
    void *UNUSED__) {
  thread_local auto socket_id = rte_socket_id();
  thread_local const auto lcore_id = rte_lcore_id();
  std::cerr << "DPDK Move thread started on lcore " + std::to_string(lcore_id) + " on socket " +
                   std::to_string(socket_id) + "\n";

  auto &lcoreInfo = dpdk_coreInfo[lcore_id];
  thread_local auto bound_to_port = lcoreInfo.port_id;
  thread_local auto bound_to_ring = lcoreInfo.ring_id;

  auto iterRelativeCoreInfo = std::find_if(dpdk_coreInfo.begin(), dpdk_coreInfo.end(), [&lcoreInfo](const auto &pair) {
    return pair.second.lcore_work_type == Rx && pair.second.port_id == bound_to_port &&
           pair.second.ring_id == bound_to_ring;
  });
  if (iterRelativeCoreInfo == dpdk_coreInfo.end()) {
    std::cerr << "No Rx core found for port " << bound_to_port << " and ring " << bound_to_ring << std::endl;
    exit(EXIT_FAILURE);
  }

  auto &moveBuffer = iterRelativeCoreInfo->second.move_buffer;
  while (!moveBuffer)
    std::this_thread::sleep_for(std::chrono::microseconds(10));

  std::vector<rte_mbuf *> unhandled_mbufs(dpdk_param.rte_mbuf_double_pointer_size_multiply * dpdk::rte_rx_burst_size);
  int unhandled_mbuf_num = 0;

  auto callback_clear = [&](dpdk::DPDKBuffer &buffer) noexcept {
    buffer.portId = bound_to_port;
    buffer.num = unhandled_mbuf_num;
    rte_memcpy(buffer.rte_mbufs.data(), unhandled_mbufs.data(), sizeof(rte_mbuf *) * unhandled_mbuf_num);
  };

  auto callback_read = [&](const MoveBuffer &buffer) noexcept {
    if (buffer.num == 0) {
      if (unhandled_mbuf_num > 0) {
        dpdk_ring_buffer->write(std::ref(callback_clear));
        unhandled_mbuf_num = 0;
      }
      return;
    }

    for (int i = 0; i < buffer.num; i++) {
      if (unhandled_mbuf_num >= unhandled_mbufs.size()) {
        dpdk_ring_buffer->write(std::ref(callback_clear));
        unhandled_mbuf_num = 0;
      }
      unhandled_mbufs[unhandled_mbuf_num++] = buffer.rte_mbufs[i];
    }
  };

  while (moveBuffer->read(std::ref(callback_read)))
    ;
  dpdk_ring_buffer->stop();
  std::cerr << "DPDK Move thread on lcore " + std::to_string(lcore_id) + " exited.\n";
  return 0;
}

int dpdk_copy(
    void *UNUSED__) {
  struct in_addr addrLocal;
  auto &lcoreInfo = dpdk_coreInfo[rte_lcore_id()];
  std::cerr << "DPDK Copy thread started on lcore " + std::to_string(rte_lcore_id()) + "\n";

  using dpdk::dpdk_local_buffer_bytes, dpdk::dpdk_move_burst;
  static_assert(dpdk_local_buffer_bytes > 0, "dpdk_write_burst must be greater than 0");
  std::array<uint8_t, dpdk_local_buffer_bytes> localBuffer;
  std::array<uint32_t, dpdk_move_burst> localOffset;
  std::array<uint16_t, dpdk_move_burst> localLength;
  std::array<openpni::process::IP, dpdk_move_burst> localAddress;
  uint16_t localIndex = 0;
  uint32_t localBytes = 0;

  dpdk::WriteParam param;
  param.length = localLength.data();
  param.offset = localOffset.data();
  param.ipPair = localAddress.data();
  param.src = localBuffer.data();
  param.burst = 0;

  auto clear = [&] noexcept {
    param.burst = localIndex;
    {
      std::shared_lock ll(dpdk_seizeMutex);
      if (dpdk_writeToMemory)
        (*dpdk_writeToMemory)(&param);
    }
    localIndex = 0;
    localBytes = 0;
  };

  auto onRead = [&](const decltype(dpdk_ring_buffer)::element_type::BufferType &buffer) noexcept {
    if (buffer.num == 0)
      return;

    openpni::process::IP address;
    for (int i = 0; i < buffer.num; i++) {
      rte_mbuf *const tempBuf = buffer.rte_mbufs[i];
      rte_ether_hdr *const ethHeader = rte_pktmbuf_mtod(tempBuf, rte_ether_hdr *);

      if (ethHeader->ether_type != rte_cpu_to_be_16(RTE_ETHER_TYPE_IPV4))
        continue;
      rte_ipv4_hdr *const ipv4Header = rte_pktmbuf_mtod_offset(tempBuf, rte_ipv4_hdr *, sizeof(rte_ether_hdr));
      if (ipv4Header->next_proto_id != IPPROTO_UDP)
        continue;

      rte_udp_hdr *const udpHeader = (rte_udp_hdr *)(ipv4Header + 1);
      const auto udpTotalLength = ntohs(udpHeader->dgram_len);
      const auto udpDatagramLength = udpTotalLength - sizeof(rte_udp_hdr);
      openpni::process::IP address;
      address.dst_addr = ntohl(ipv4Header->dst_addr);
      address.dst_port = ntohs(udpHeader->dst_port);
      address.src_addr = ntohl(ipv4Header->src_addr);
      address.src_port = ntohs(udpHeader->src_port);
      // printf("%u:%u -> %u:%u\n", ipv4Header->src_addr, udpHeader->src_port,
      // ipv4Header->dst_addr, udpHeader->dst_port);

      if (localIndex == dpdk_move_burst || localBytes + udpDatagramLength > localBuffer.size())
        clear();

      localAddress[localIndex] = address;
      localLength[localIndex] = udpDatagramLength;
      localOffset[localIndex] = localBytes;
      rte_memcpy(&localBuffer[localOffset[localIndex]], udpHeader + 1, localLength[localIndex]);

      localIndex++;
      localBytes += udpDatagramLength;
    }
    if (localIndex != 0)
      clear();
    rte_pktmbuf_free_bulk((rte_mbuf **)buffer.rte_mbufs.data(), buffer.num);
    dpdk_buffer_free[buffer.portId] += buffer.num;
  };
  while (dpdk_ring_buffer->read(std::ref(onRead)))
    ;
  std::cerr << "DPDK Copy thread on lcore " + std::to_string(rte_lcore_id()) + " exited.\n";
  return 0;
}
int dpdk_thread(
    void *__arg) {
  const auto lcore_id = rte_lcore_id();
  switch (dpdk_coreInfo[lcore_id].lcore_work_type) {
  case Rx:
    // printf("Rx in core %d\n", lcore_id);
    dpdk_rx(nullptr);
    break;
  case Move:
    dpdk_move(nullptr);
  case Copy:
    // printf("Move in core %d\n", lcore_id);
    dpdk_copy(nullptr);
  default:
    break;
  }
  return 0;
}
int port_init(
    uint16_t port, uint8_t rx_rings) {
  using namespace openpni;
  struct rte_eth_conf port_conf;
  uint16_t nb_rxd = dpdk::rte_rx_ring_size;
  uint16_t nb_txd = dpdk::rte_tx_ring_size;
  int retval;
  uint16_t q;
  struct rte_eth_dev_info dev_info;
  struct rte_eth_txconf txconf;

  if (!rte_eth_dev_is_valid_port(port))
    return -1;

  memset(&port_conf, 0, sizeof(struct rte_eth_conf));
  // port_conf.rxmode.offloads &= ~RTE_ETH_RX_OFFLOAD_CHECKSUM;
  // port_conf.rxmode.offloads &= ~RTE_ETH_RX_OFFLOAD_IPV4_CKSUM;
  // port_conf.rxmode.offloads &= ~RTE_ETH_RX_OFFLOAD_UDP_CKSUM;
  // port_conf.rxmode.offloads &= ~RTE_ETH_RX_OFFLOAD_TCP_CKSUM;

  retval = rte_eth_dev_info_get(port, &dev_info);
  if (retval != 0) {
    return retval;
  }

  if (dev_info.tx_offload_capa & RTE_ETH_TX_OFFLOAD_MBUF_FAST_FREE)
    port_conf.txmode.offloads |= RTE_ETH_TX_OFFLOAD_MBUF_FAST_FREE;

  /* Configure the Ethernet device. */
  retval = rte_eth_dev_configure(port, rx_rings, dpdk::tx_rings, &port_conf);
  if (retval != 0)
    return retval;

  retval = rte_eth_dev_adjust_nb_rx_tx_desc(port, &nb_rxd, &nb_txd);
  if (retval != 0)
    return retval;

  /* Allocate and set up 1 RX queue per Ethernet port. */
  for (q = 0; q < rx_rings; q++) {
    retval = rte_eth_rx_queue_setup(port, q, nb_rxd, rte_eth_dev_socket_id(port), NULL, dpdk_mempool[port]);
    if (retval < 0)
      return retval;
  }

  txconf = dev_info.default_txconf;
  txconf.offloads = port_conf.txmode.offloads;
  /* Allocate and set up 1 TX queue per Ethernet port. */
  for (q = 0; q < dpdk::tx_rings; q++) {
    retval = rte_eth_tx_queue_setup(port, q, nb_txd, rte_eth_dev_socket_id(port), &txconf);
    if (retval < 0)
      return retval;
  }

  /* Starting Ethernet port. 8< */
  retval = rte_eth_dev_start(port);
  /* >8 End of starting of ethernet port. */
  if (retval < 0)
    return retval;

  /* Display the port MAC address. */
  struct rte_ether_addr addr;
  retval = rte_eth_macaddr_get(port, &addr);
  dpdk_port_info[port].mac = addr;
  if (retval != 0)
    return retval;

  /* Enable RX in promiscuous mode for the Ethernet device. */
  retval = rte_eth_promiscuous_enable(port);
  /* End of setting RX port in promiscuous mode. */
  if (retval != 0)
    return retval;

  return 0;
}
static std::string numaNodeName(
    int socket_id) {
  if (socket_id == -1)
    return "NUMA_ANY";
  return "NUMA" + std::to_string(socket_id);
}
int dpdk_init(
    dpdk::DPDKParam __p, openpni::process::FuncLogger &__f) {
  dpdk_param = __p;
  using namespace openpni;

  uint16_t portid;

  /* Initializion the Environment Abstraction Layer (EAL). 8< */
  const char *eal_args[]{
      "dpdk",
      // "--in-memory",
      // "--iova-mode",
      // "va",
      // "--socket-mem", "0,1024",
      // "--socket-limit", "0,1024"
  };
  static int ret_rte_eal_init;
  ret_rte_eal_init = rte_eal_init(sizeof(eal_args) / sizeof(eal_args[0]), (char **)eal_args);
  if (ret_rte_eal_init < 0) {
    __f("Failed to init dpdk runtime environment.");
    throw openpni::exceptions::invalid_environment();
  } /* >8 End of initialization the Environment Abstraction Layer (EAL). */

  /* Count how many socket int the system. */
  __f("There are " + std::to_string(rte_socket_count()) + " sockets detected.");
  if (rte_socket_count() > dpdk::rte_max_supported_socket_num) {
    __f("The socket num is more than supported. Expected " + std::to_string(dpdk::rte_max_supported_socket_num) +
        ", get " + std::to_string(rte_socket_count()) + ".");
    throw openpni::exceptions::resource_unavailable();
  }

  /* Count how many ports in each socket */
  std::map<int, std::set<int>> socket_port_map;
  RTE_ETH_FOREACH_DEV(portid) {
    const auto socketIdOfPort = rte_eth_dev_socket_id(portid);
    __f("Port " + std::to_string(portid) + " is on socket " + numaNodeName(socketIdOfPort) + ".");
    if (socket_port_map.find(socketIdOfPort) == socket_port_map.end()) {
      socket_port_map[socketIdOfPort] = std::set<int>();
    }
    socket_port_map[socketIdOfPort].insert(portid);
  }

  for (const auto &[socket_id, port_set] : socket_port_map) {
    const auto socketIdOfPort = rte_eth_dev_socket_id(portid);
    const auto portNumOfSocket = port_set.size();
    auto rte_pktmbuf_pool_size = dpdk::rte_pktmbuf_pool_size / portNumOfSocket;
    int trys;
    for (trys = 0; trys < dpdk::rte_malloc_retry_num; trys++) {
      __f("Allocating MBUF pool...");
      std::vector<rte_mempool *> allocated_pools;

      for (const auto &port_id : port_set) {
        auto new_pool = rte_pktmbuf_pool_create(std::string("DPDK_MBUF_FOR_PORT_" + std::to_string(port_id)).c_str(),
                                                rte_pktmbuf_pool_size, dpdk::rte_mbuf_cache_size, 0,
                                                RTE_MBUF_DEFAULT_BUF_SIZE, socket_id);
        if (new_pool == nullptr) {
          __f("Failed to allocate MBUF pool for port " + std::to_string(port_id) + ".");
          for (auto &pool : allocated_pools) {
            rte_mempool_free(pool);
          }
          break;
        } else {
          dpdk_mempool[port_id] = new_pool;
          allocated_pools.push_back(new_pool);
          dpdk_buffer_free[port_id] = rte_pktmbuf_pool_size;
          dpdk_buffer_total[port_id] = rte_pktmbuf_pool_size;
        }
      }
      if (allocated_pools.size() == port_set.size()) {
        __f("Allocated MBUF pool for all ports on " + numaNodeName(socket_id) + ".");
        break;
      }
      __f("Failed. Acquiring half and retry...");
      rte_pktmbuf_pool_size /= 2;
    }
    if (trys == dpdk::rte_malloc_retry_num) {
      __f("Failed to create RTE memory pool.");
      throw openpni::exceptions::resource_unavailable();
    }
  }
  /* >8 End of allocating mempool to hold mbuf. */

  /* Check that the number of ports to send/receive on. */
  using openpni::process::dpdk::rte_port_num;
  rte_port_num = rte_eth_dev_count_avail();
  __f("There are " + std::to_string(rte_port_num) + " ports detected.");
  if (__p.etherIpBind.size() != rte_port_num) {
    __f("Instead, " + std::to_string(__p.etherIpBind.size()) + " ports is supposed, it is not compatible.");
    throw openpni::exceptions::invalid_environment();
  }

  /* Initializing all ports. 8< */
  RTE_ETH_FOREACH_DEV(portid)
  if (port_init(portid, __p.rxThreadNumForEachPort) != 0) {
    __f("Failed to init port " + std::to_string(portid) + ".");
    throw openpni::exceptions::resource_unavailable();
  }
  /* >8 End of initializing all ports. */

  __f("There are " + std::to_string(rte_port_num) + " ports used.");

  if (rte_port_num < 1) {
    __f("The is no available network port.");
    throw openpni::exceptions::resource_unavailable();
  } else if (rte_port_num >= dpdk::rte_max_supported_port_num) {
    __f("The port num is more than supported. Expected " + std::to_string(dpdk::rte_max_supported_port_num) + ", get " +
        std::to_string(rte_port_num) + ".");
    throw openpni::exceptions::resource_unavailable();
  }

  int lcores = rte_lcore_count() - 1;
  __f("Calling for " + std::to_string(__p.rxThreadNumForEachPort) + " RX threads for each port, " +
      std::to_string(__p.copyThreadNum) + " copy threads.");

  dpdk_ring_buffer = std::make_unique<decltype(dpdk_ring_buffer)::element_type>(
      std::size_t(dpdk::rte_pktmbuf_pool_size + 1) / dpdk::rte_rx_burst_size /
          __p.rte_mbuf_double_pointer_size_multiply * __p.rte_mbuf_double_pointer_num_pultiply,
      [&](decltype(dpdk_ring_buffer)::element_type::BufferType &buffer) -> void {
        buffer.rte_mbufs.resize(dpdk::rte_rx_burst_size * __p.rte_mbuf_double_pointer_size_multiply);
      });

  // Init threads:
  // Init Rx/Move threads for each port:
  std::set<unsigned> usedLcores;
  {
    RTE_ETH_FOREACH_DEV(portid) {
      const auto socketId = rte_eth_dev_socket_id(portid);
      const bool numa_any = socketId == -1;
      unsigned lcore_id;
      unsigned foundRxCores = 0;
      unsigned foundMoveCores = 0;
      RTE_LCORE_FOREACH_WORKER(lcore_id) {
        if (usedLcores.contains(lcore_id))
          continue;
        if (rte_lcore_to_socket_id(lcore_id) != socketId && !numa_any)
          continue;

        dpdk_coreInfo[lcore_id] = LCoreInfo();
        dpdk_coreInfo[lcore_id].lcore_work_type = Rx;
        dpdk_coreInfo[lcore_id].ipv4 = dpdk::ipString2Int(__p.etherIpBind[portid]);
        dpdk_coreInfo[lcore_id].port_id = portid;
        dpdk_coreInfo[lcore_id].ring_id = foundRxCores;
        __f("Port " + std::to_string(portid) + " is bound to Rx lcore " + std::to_string(lcore_id) + ".");

        usedLcores.insert(lcore_id);

        foundRxCores++;
        if (foundRxCores >= __p.rxThreadNumForEachPort)
          break;
      }
      if (foundRxCores != __p.rxThreadNumForEachPort) {
        __f("No available core for port " + std::to_string(portid) + ".");
        throw openpni::exceptions::resource_unavailable();
      }

      RTE_LCORE_FOREACH_WORKER(lcore_id) {
        if (usedLcores.contains(lcore_id))
          continue;
        if (rte_lcore_to_socket_id(lcore_id) != socketId && !numa_any)
          continue;

        dpdk_coreInfo[lcore_id] = LCoreInfo();
        dpdk_coreInfo[lcore_id].lcore_work_type = Move;
        dpdk_coreInfo[lcore_id].port_id = portid;
        dpdk_coreInfo[lcore_id].ring_id = foundMoveCores;
        __f("Lcore " + std::to_string(lcore_id) + " is bound to Move thread for port " + std::to_string(portid) + ".");
        usedLcores.insert(lcore_id);
        foundMoveCores++;
        if (foundMoveCores >= __p.rxThreadNumForEachPort)
          break;
      }
      if (foundMoveCores != __p.rxThreadNumForEachPort) {
        __f("No available core for Move thread of port " + std::to_string(portid) + ".");
        throw openpni::exceptions::resource_unavailable();
      }
    }
  }
  total_rx_thread_num = __p.rxThreadNumForEachPort * rte_port_num;

  // Init Copy threads for each copy thread:
  {
    unsigned lcore_id;
    unsigned lcore_copy = 0;
    RTE_LCORE_FOREACH_WORKER(lcore_id) {
      if (usedLcores.contains(lcore_id))
        continue;
      dpdk_coreInfo[lcore_id] = LCoreInfo();
      dpdk_coreInfo[lcore_id].lcore_work_type = Copy;
      usedLcores.insert(lcore_id);
      lcore_copy++;
      if (lcore_copy >= __p.copyThreadNum)
        break;
    }
  }

  __f("There are " + std::to_string(lcores) + " cores available, " + std::to_string(usedLcores.size()) + " in use.");

  for (auto &[lcore_id, info] : dpdk_coreInfo) {
    rte_eal_remote_launch(dpdk_thread, nullptr, lcore_id);
  }

  std::atexit([] {
    /* clean up the EAL */
    dpdk_stopFlag = true;
    rte_eal_mp_wait_lcore();
    rte_eal_cleanup();
  });

  return 0;
}

void dpdk::InitDPDK(
    DPDKParam __p, process::FuncLogger &&__f) {
  process::FuncLogger logger = __f;
  static std::once_flag dpdk_init_once;
  std::call_once(dpdk_init_once, [&] { dpdk_init(__p, __f); });
}

static std::binary_semaphore g_semaphore_dpdk_resource{1};
// uint64_t dpdk::getDPDKBufferNum()
// {
//     return dpdk_buffer_total;
// }
// uint64_t dpdk::getDPDKBufferAvailable()
// {
//     return dpdk_buffer_free;
// }
uint16_t dpdk::getDPDKMoveThreadNum() {
  uint16_t result{0};
  for (const auto &[key, value] : dpdk_coreInfo)
    if (value.lcore_work_type == Move)
      result++;
  return result;
}
bool dpdk::seizeDPDK(
    process::FuncLogger &__f, FuncWriteToMemory &__writeToMemory) {
  if (g_semaphore_dpdk_resource.try_acquire()) {
    {
      std::lock_guard ll(dpdk_seizeMutex);
      dpdk_writeToMemory = &__writeToMemory;
    }
    __f("Seize DPDK successfully.");
    return true;
  } else {
    __f("Failed to seize DPDK.");
    throw openpni::exceptions::resource_unavailable();
  }
}

void dpdk::releaseDPDK() {
  g_semaphore_dpdk_resource.release();
  {
    std::lock_guard ll(dpdk_seizeMutex);
    dpdk_writeToMemory = nullptr;
  }
}

#endif