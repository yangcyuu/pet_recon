#pragma once
#ifndef PNI_STANDARD_ALGORITHM_AQUISITION_IMPL_SOCKET
#define PNI_STANDARD_ALGORITHM_AQUISITION_IMPL_SOCKET
#include <arpa/inet.h>
#include <chrono>
#include <event2/event.h>
#include <event2/util.h>
#include <map>
#include <netinet/in.h>
#include <shared_mutex>
#include <sys/socket.h>
#include <sys/types.h>
#include <thread>
#include <unistd.h>

#include "IPSetting.hpp"
#include "TimedDataBuffer.hpp"
#include "include/process/Acquisition.hpp"

struct event_base;
struct event;
namespace openpni::process::socket {
class LibEventThreadBuffer;
class Socket::SocketImpl {
public:
  SocketImpl(AcquisitionInfo, process::FuncLogger &&);
  virtual ~SocketImpl();
  SocketImpl(const SocketImpl &) = delete;
  SocketImpl &operator=(const SocketImpl &) = delete;

public:
  Status status() noexcept;
  bool start();
  bool stop() noexcept;
  bool isFinished() noexcept;
  std::optional<RawDataView> read() noexcept;

private:
  bool allocate();
  bool startThreads();
  bool bind();

private: // thread local
  void threadSocket(int threadId);
  void threadMark();

private:
  struct ThreadCtx;
  static void on_udp_read(evutil_socket_t fd, short events, void *arg);
  static void on_tick(evutil_socket_t fd, short events, void *arg);

private:
  struct SocketRuntimePair {
    evutil_socket_t fd{-1};
    event_base *base{nullptr};
    event *ev_read{nullptr};
    event *ev_tick{nullptr}; // 周期刷新（把未满 burst 的数据刷入）
    SocketRuntimePair() = default;
    ~SocketRuntimePair() = default;
  };
  struct AcquisitionLoop {
    bool m_stopped{false};
    bool m_finished{false};
    std::unique_ptr<SocketRuntimePair[]> m_socketRuntimes;
    std::vector<std::jthread> m_threadsKernelData;
    std::jthread m_threadMark;
    uint64_t m_lastReadingIndex{0};
    std::optional<RawDataView> m_lastReadResult;
    uint64_t m_lastReadoutCount{0};
    std::atomic<uint64_t> m_wrongPacketCount{0};

    ~AcquisitionLoop() { m_threadsKernelData.clear(); }
  };

private:
  const AcquisitionInfo mc_param;
  std::unique_ptr<timed_buffer::Groups> m_buffer;

  bool m_status{true};
  process::FuncLogger m_logger;
  std::function<uint16_t(IP, uint8_t *, uint16_t)> m_ipMapper;
  std::unique_ptr<AcquisitionLoop> m_runtime;
};
} // namespace openpni::process::socket

#endif // !PNI_STANDARD_ALGORITHM_AQUISITION_IMPL_KERNEL