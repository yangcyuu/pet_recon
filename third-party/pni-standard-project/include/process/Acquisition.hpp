#pragma once
#ifndef PNI_STANDARD_ALGORITHM_AQUISITION_INCLUDE
#define PNI_STANDARD_ALGORITHM_AQUISITION_INCLUDE
#include <concepts>
#include <coroutine>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "../Exceptions.hpp"
#include "../PnI-Config.hpp"
#if PNI_STANDARD_CONFIG_ENABLE_DPDK
#include <rte_ethdev.h>
#endif

namespace openpni::process {
/** 这个数据结构指示了采集算法的标准输出。
 *  并且假定：
 *      1. 通道与探测器一一对应。
 *      2. 探测器通过UDP数据包向主机发送数据。
 *      3. UDP数据包必须预设最大大小与最小大小。
 *      4. 必须保证数据在内存中。
 *      5. 无需保证数据包在内存中是紧密排列的。
 */

struct RawDataView // 指针不要在外面析构！
{
  // 以下是数据信息
  uint8_t *data;           // 存放数据的地方，允许间断但不允许重叠
  uint16_t *length;        // 每个数据包的长度
  uint64_t *offset;        // 每个数据包的起始字节偏移量 offset[0] ?= 0
  uint16_t *channel;       // 每个数据包来自哪个通道，正常范围[0,N)，允许出现65535作为异常值
  uint64_t count{0};       // 总数据包数量
  uint64_t clock_ms{0};    // 这些数据的起始计算机时钟（单位：毫秒）
  uint32_t duration_ms{0}; // 这些数据的持续时长（单位：毫秒）
  // 以下是不可变信息
  uint16_t channelNum{0}; // 有效通道数量
};
struct AcquisitionInfo {
  struct ChannelSetting {
    uint32_t ipSource;        // 源IP地址，为0表示ANY
    uint16_t portSource;      // 源端口号，为0表示ANY
    uint32_t ipDestination;   // 目的IP地址
    uint16_t portDestination; // 目的端口号
    uint16_t channelIndex{
        0}; // 通道号，自定义顺序，但是要求范围在[0,totalChannelNum)范围(可聚合)，一般不使用65535（后者为特殊标志）
    std::function<bool(uint8_t *__udpDatagram, uint16_t __udpLegnth, uint32_t __ipSource, uint16_t __portSource)>
        quickFilter{nullptr}; // 过滤器，若返回false则丢弃该数据包或将其通道设置为-1；若不设置过滤器则不过滤
  };
  std::vector<ChannelSetting> channelSettings; // 通道地址及端口
  uint16_t storageUnitSize{1024};              // 指定一个UDP包使用多少字节进行存储，不小于可用UDP包的最小大小
  uint64_t maxBufferSize{1024 * 1024 * 1024};  // 指定内存缓存区的大小，分配数额不会小于该值
  uint16_t timeSwitchBuffer_ms{100};           // 每次读出数据大约为多少毫秒
  uint16_t totalChannelNum{0};                 // 采集通道数量
};

// 协程任务类，用于辅助从采集算法中读取数据
class AcquisitionHandler {
public:
  struct promise_type {
    std::optional<RawDataView> currentValue;
    AcquisitionHandler get_return_object() noexcept {
      return AcquisitionHandler{std::coroutine_handle<promise_type>::from_promise(*this)};
    }
    std::suspend_always initial_suspend() noexcept { return {}; }
    std::suspend_always final_suspend() noexcept { return {}; }
    std::suspend_always yield_value(
        std::optional<RawDataView> value) noexcept {
      currentValue = std::move(value);
      return {};
    }
    void return_void() noexcept { currentValue.reset(); }
    void unhandled_exception() noexcept { std::terminate(); }
  };

public:
  AcquisitionHandler(
      std::coroutine_handle<promise_type> h) noexcept
      : handle(h) {}
  ~AcquisitionHandler() noexcept {
    if (handle)
      handle.destroy();
  }
  AcquisitionHandler(const AcquisitionHandler &) = delete;
  AcquisitionHandler &operator=(const AcquisitionHandler &) = delete;

  std::optional<RawDataView> get() const noexcept { return handle.promise().currentValue; }

private:
  std::coroutine_handle<promise_type> handle;
};

template <typename AA>
AcquisitionHandler read_from_acquisition(
    std::reference_wrapper<AA> __acquisition) noexcept {
  while (!__acquisition.get().isFinished()) {
    if (const auto data = __acquisition.get().read(); data) {
      co_yield *data;
    } else
      std::this_thread::yield();
  }
  co_return;
}

using FuncLogger = std::function<void(const std::string &)>;
/** 采集算法实例，有三种具体实现
 *  Socket：使用内核协议栈实现的算法，需安装libevent
 *  DPDK：使用DPDK进行实现的算法，需安装DPDK
 *  DPU：使用DPU进行实现的算法，需安装DPU
 */
namespace socket {
struct Status {
  uint64_t volume;         // 内存池总量
  uint64_t used;           // 内存池使用量
  uint64_t totalRxPackets; // 总接收包数
  uint64_t totalRxBytes;   // 总接收字节数
  uint64_t unknown;        // 未知/错误地址或错误长度包数
};
class Socket {
public:
  Socket(AcquisitionInfo, FuncLogger &&) noexcept;
  ~Socket() noexcept;

public:
  Status status() noexcept;
  bool start();
  bool stop() noexcept;
  bool isFinished() noexcept;
  std::optional<RawDataView> read() noexcept;

private:
  class SocketImpl;
  std::unique_ptr<SocketImpl> impl;
};

} // namespace socket

#if PNI_STANDARD_CONFIG_ENABLE_DPDK
namespace dpdk {

struct Status {
  uint64_t volume;         // 内存池总量
  uint64_t used;           // 内存池使用量
  uint64_t totalRxPackets; // 总接收包数
  uint64_t totalRxBytes;   // 总接收字节数
  uint64_t unknown;        // 未知/错误地址或错误长度包数
};

struct DPDKParam {
  std::vector<std::string> etherIpBind;
  uint8_t copyThreadNum;
  uint8_t rxThreadNumForEachPort{1};
  uint16_t rte_mbuf_double_pointer_size_multiply{32};
  uint16_t rte_mbuf_double_pointer_num_pultiply{2};
};
void InitDPDK(DPDKParam, FuncLogger &&);
class DPDK {
public:
  DPDK(AcquisitionInfo, FuncLogger &&) noexcept;
  ~DPDK() noexcept;

public:
  Status status() noexcept;
  bool start();
  bool stop() noexcept;
  bool isFinished() noexcept;
  std::optional<RawDataView> read() noexcept;

private:
  class DPDKImpl;
  std::unique_ptr<DPDKImpl> impl;
};

// Internal status for each port
struct PortStatus {
  rte_eth_stats rte_stats;
  uint64_t dpdkBufferFree;   // DPDK包描述符可用量
  uint64_t dpdkBufferVolume; // DPDK包描述符总量
};
inline constexpr uint8_t rte_max_supported_port_num{64};
extern uint8_t rte_port_num;
extern PortStatus dpdk_status[rte_max_supported_port_num];
} // namespace dpdk
#endif
#if PNI_STANDARD_CONFIG_ENABLE_DPU
namespace dpu {
std::unique_ptr<AcquisitionAlgorithm> DPU(AcquisitionParameter, common::FuncLogger && = common::FuncLogger());
}
#endif

}; // namespace openpni::process

#endif // !PNI_STANDARD_ALGORITHM_AQUISITION_INCLUDE
