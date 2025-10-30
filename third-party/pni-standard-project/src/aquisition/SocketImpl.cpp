#include "SocketImpl.hpp"

#include <iostream>
#include <memory>
#include <ranges>
#include <sstream>

#include "Consts.hpp"
#include "src/common/TemplateFunctions.hpp"

#define CALL_IF(AAA)                                                                     \
  if (m_logger)                                                                          \
    m_logger(AAA);
#define AS_STRING(expression) ((std::stringstream() << expression).str())

namespace c = openpni::consts;
namespace cro = std::chrono;

static constexpr uint64_t burstSize = 64;

static std::string ipInt2Str(uint32_t ip) noexcept {
  return std::to_string((ip >> 24) & 0xFF) + "." + std::to_string((ip >> 16) & 0xFF) +
         "." + std::to_string((ip >> 8) & 0xFF) + "." + std::to_string(ip & 0xFF);
}

namespace openpni::process::socket {
// 基于libevent实现
struct Socket::SocketImpl::ThreadCtx {
  Socket::SocketImpl *impl;
  int threadId;

  uint32_t dst_addr_host;
  uint16_t dst_port;

  const size_t storageUnitSize;
  const size_t bufferSize;

  std::unique_ptr<uint8_t[]> tempBuffer;
  std::unique_ptr<uint16_t[]> tempChannel;
  std::unique_ptr<uint16_t[]> tempLength;
  std::unique_ptr<uint32_t[]> tempOffset;

  uint32_t bufferOffset{0};
  uint32_t bufferIndex{0};

  sockaddr_in sender{};
  socklen_t senderLen{sizeof(sender)};

  ThreadCtx(Socket::SocketImpl *p, int tid, uint32_t dst_addr, uint16_t dst_port_,
            size_t unit)
      : impl(p), threadId(tid), dst_addr_host(dst_addr), dst_port(dst_port_),
        storageUnitSize(unit), bufferSize(unit * burstSize + 8192),
        tempBuffer(std::make_unique<uint8_t[]>(bufferSize)),
        tempChannel(std::make_unique<uint16_t[]>(burstSize)),
        tempLength(std::make_unique<uint16_t[]>(burstSize)),
        tempOffset(std::make_unique<uint32_t[]>(burstSize)) {
    std::memset(&sender, 0, sizeof(sender));
  }

  void flush_if_needed(bool force = false) {
    if (!bufferIndex)
      return;
    if (!force && bufferIndex < burstSize)
      return;

    if (!timed_buffer::write(*impl->m_buffer, &tempBuffer[0], &tempLength[0],
                             &tempChannel[0], &tempOffset[0], bufferIndex)) {
      impl->m_buffer->writeThreadGo.wait(false);
    }
    bufferIndex = 0;
    bufferOffset = 0;
  }
};

// 回调：UDP 数据包
void Socket::SocketImpl::on_udp_read(evutil_socket_t fd, short, void *arg) {
  auto *ctx = reinterpret_cast<ThreadCtx *>(arg);
  auto *impl = ctx->impl;

  constexpr size_t kMaxUdp = 65536;
  if (ctx->bufferSize - ctx->bufferOffset < kMaxUdp) {
    ctx->flush_if_needed(true);
  }

  uint8_t *recvPtr = &ctx->tempBuffer[ctx->bufferOffset];
  int len = ::recvfrom(fd, recvPtr, static_cast<int>(ctx->bufferSize - ctx->bufferOffset),
                       0, reinterpret_cast<sockaddr *>(&ctx->sender), &ctx->senderLen);
  if (len <= 0)
    return;
  if (len > impl->mc_param.storageUnitSize)
    return;

  IP ipPairRuntime{};
  ipPairRuntime.src_addr = ntohl(ctx->sender.sin_addr.s_addr);
  ipPairRuntime.src_port = ntohs(ctx->sender.sin_port);
  ipPairRuntime.dst_addr = ctx->dst_addr_host;
  ipPairRuntime.dst_port = ctx->dst_port;

  const auto channelIndex =
      impl->m_ipMapper(ipPairRuntime, recvPtr, static_cast<uint16_t>(len));
  if (channelIndex >= impl->mc_param.totalChannelNum) {
    impl->m_runtime->m_wrongPacketCount++;
    return;
  }

  // ctx->tempChannel[ctx->bufferIndex] = static_cast<uint16_t>(channelIndex);
  // ctx->tempOffset[ctx->bufferIndex] = ctx->bufferOffset;
  // ctx->tempLength[ctx->bufferIndex] = static_cast<uint16_t>(len);

  // ctx->bufferIndex++;
  // ctx->bufferOffset += static_cast<uint32_t>(len);
  // 复制 n 次
  constexpr int maxCopyCount = 1;
  for (int i = 0; i < maxCopyCount; ++i) {
    if (ctx->bufferIndex == burstSize) {
      ctx->flush_if_needed(true);
    }
    if (ctx->bufferSize - ctx->bufferOffset < len) {
      ctx->flush_if_needed(true);
    }

    uint8_t *currentRecvPtr = &ctx->tempBuffer[ctx->bufferOffset];
    if (i > 0) { // 第一次接收时数据已在正确位置
      std::memcpy(currentRecvPtr, recvPtr, len);
    }

    ctx->tempChannel[ctx->bufferIndex] = static_cast<uint16_t>(channelIndex);
    ctx->tempOffset[ctx->bufferIndex] = ctx->bufferOffset;
    ctx->tempLength[ctx->bufferIndex] = static_cast<uint16_t>(len);

    ctx->bufferIndex++;
    ctx->bufferOffset += static_cast<uint32_t>(len);
  }

  if (ctx->bufferIndex == burstSize) {
    ctx->flush_if_needed(true);
  }
}

// 回调：定时器 tick
void Socket::SocketImpl::on_tick(evutil_socket_t, short, void *arg) {
  auto *ctx = reinterpret_cast<ThreadCtx *>(arg);
  if (ctx->impl->m_runtime->m_stopped)
    return;
  ctx->flush_if_needed(true);
}
}; // namespace openpni::process::socket

namespace openpni::process::socket {
Socket::SocketImpl::SocketImpl(AcquisitionInfo __param, process::FuncLogger &&__logger)
    : mc_param(__param), m_logger(__logger),
      m_ipMapper(toIPMapper(mc_param.channelSettings)) {
  CALL_IF(AS_STRING("Trying to allocate esitmated "
                    << (double(mc_param.maxBufferSize) / c::GB) << " GB memory."))
  if (!allocate())
    m_status = false;
}

Socket::SocketImpl::~SocketImpl() {}

Status Socket::SocketImpl::status() noexcept {
  Status result;
  result.totalRxBytes = m_buffer->totalRxBytes;
  result.totalRxPackets = m_buffer->writingIndex;
  result.volume = m_buffer->realAllocatePacketNum;
  result.used = result.volume - m_buffer->empty;
  result.unknown = m_runtime ? m_runtime->m_wrongPacketCount.load() : 0;
  return result;
}

bool Socket::SocketImpl::start() {
  if (!m_status)
    return false;
  if (m_runtime) {
    timed_buffer::clear(*m_buffer);
  }
  m_runtime = std::make_unique<decltype(m_runtime)::element_type>();

  CALL_IF("Trying to bind udp sockets.")
  m_status = m_status && bind();
  m_buffer->clockStart = std::chrono::steady_clock::now();

  CALL_IF("Trying to start aquisition threads.")
  startThreads();
  CALL_IF("Aquisition started.")
  return m_status;
}

bool Socket::SocketImpl::stop() noexcept {
  if (!m_runtime)
    return true;
  m_runtime->m_stopped = true;

  const size_t N = mc_param.channelSettings.size();
  for (size_t i = 0; i < N; ++i) {
    auto &rt = m_runtime->m_socketRuntimes[i];
    if (rt.base) {
      event_base_loopbreak(rt.base);
    }
  }

  timed_buffer::stop(*m_buffer);
  return true;
}

bool Socket::SocketImpl::isFinished() noexcept {
  return m_runtime ? m_runtime->m_finished : true;
}

std::optional<RawDataView> Socket::SocketImpl::read() noexcept {
  if (!m_runtime)
    return std::nullopt;

  if (m_runtime->m_lastReadoutCount) {
    m_buffer->empty += m_runtime->m_lastReadoutCount;
    m_buffer->writeThreadGo.test_and_set();
    m_buffer->writeThreadGo.notify_all();
  }
  if (m_runtime->m_lastReadResult.has_value()) {
    decltype(m_runtime->m_lastReadResult) result;
    std::swap(result, m_runtime->m_lastReadResult);
    m_runtime->m_lastReadoutCount = result->count;
    return result;
  }

  auto readMark = timed_buffer::read(*m_buffer);
  if (readMark) {
#ifndef PNI_USE_CAS_LOOP
    std::lock_guard lll(m_buffer->smutex);
#endif
    const auto indexEnd = readMark->indexEnd;
    RawDataView result;
    const auto [lastReadingRound, lastReadingIndexModded] =
        std::lldiv(m_runtime->m_lastReadingIndex, m_buffer->realAllocatePacketNum);
    const auto [thisReadingRound, thisReadingIndexModded] =
        std::lldiv(indexEnd, m_buffer->realAllocatePacketNum);
    if (thisReadingRound == lastReadingRound) {
      result.channel = &m_buffer->channel[lastReadingIndexModded];
      result.channelNum = mc_param.totalChannelNum;
      result.count =
          (indexEnd - m_runtime->m_lastReadingIndex) % m_buffer->realAllocatePacketNum;
      if (result.count) {
        result.clock_ms = m_buffer->clock[lastReadingIndexModded];
        result.duration_ms =
            m_buffer->clock[lastReadingIndexModded + result.count - 1] - result.clock_ms;
      } else {
        result.clock_ms = readMark->clockEnd - readMark->duration;
        result.duration_ms = readMark->duration;
      }
      result.data = &m_buffer->data[0];
      result.length = &m_buffer->length[lastReadingIndexModded];
      result.offset = &m_buffer->offset[lastReadingIndexModded];
    } else {
      result.channel = &m_buffer->channel[lastReadingIndexModded];
      result.channelNum = mc_param.totalChannelNum;
      result.clock_ms = m_buffer->clock[lastReadingIndexModded];
      result.count = m_buffer->realAllocatePacketNum - lastReadingIndexModded;
      result.duration_ms =
          result.count
              ? (m_buffer->clock[m_buffer->realAllocatePacketNum - 1] - result.clock_ms)
              : 0;
      result.data = &m_buffer->data[0];
      result.length = &m_buffer->length[lastReadingIndexModded];
      result.offset = &m_buffer->offset[lastReadingIndexModded];

      m_runtime->m_lastReadResult = RawDataView();
      m_runtime->m_lastReadResult->channel = &m_buffer->channel[0];
      m_runtime->m_lastReadResult->channelNum = mc_param.totalChannelNum;
      m_runtime->m_lastReadResult->count = indexEnd % m_buffer->realAllocatePacketNum;
      m_runtime->m_lastReadResult->clock_ms = m_buffer->clock[0];
      m_runtime->m_lastReadResult->duration_ms =
          m_runtime->m_lastReadResult->count
              ? (m_buffer->clock[m_runtime->m_lastReadResult->count - 1] -
                 m_runtime->m_lastReadResult->clock_ms)
              : 0;
      m_runtime->m_lastReadResult->data = &m_buffer->data[0];
      m_runtime->m_lastReadResult->length = &m_buffer->length[0];
      m_runtime->m_lastReadResult->offset = &m_buffer->offset[0];
    }
    m_runtime->m_lastReadingIndex = indexEnd;
    m_runtime->m_lastReadoutCount = result.count;
    return result;
  } else {
    m_runtime->m_finished = true;
    m_runtime->m_lastReadoutCount = 0;
    return std::nullopt;
  }
}

bool Socket::SocketImpl::allocate() {
  m_buffer = timed_buffer::makeGroups(mc_param.maxBufferSize / mc_param.storageUnitSize,
                                      mc_param.storageUnitSize, burstSize,
                                      mc_param.channelSettings.size());
  if (!m_buffer) {
    CALL_IF("Failed to allocate memory.")
    return false;
  }
  CALL_IF("Allocate finished.")
  return true;
}

bool Socket::SocketImpl::startThreads() {
  for (const auto threadId : std::views::iota(0ull, mc_param.channelSettings.size()))
    m_runtime->m_threadsKernelData.emplace_back(std::mem_fn(&SocketImpl::threadSocket),
                                                this, static_cast<int>(threadId));

  m_runtime->m_threadMark = std::jthread(std::mem_fn(&SocketImpl::threadMark), this);
  CALL_IF("Threads started.")
  return true;
}

bool Socket::SocketImpl::bind() {
  bool success = true;

  const size_t N = mc_param.channelSettings.size();
  m_runtime->m_socketRuntimes = std::make_unique<SocketRuntimePair[]>(N);

  for (size_t i = 0; i < N; ++i) {
    const auto &ip = mc_param.channelSettings[i];
    auto &rt = m_runtime->m_socketRuntimes[i];

    rt.fd = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (rt.fd < 0) {
      CALL_IF("Failed to create UDP socket.");
      success = false;
      continue;
    }

    evutil_make_socket_nonblocking(rt.fd);

    int reuse = 1;
    setsockopt(rt.fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
    int rcvbuf = 1024 * 1024 * 80; // 80 MiB
    setsockopt(rt.fd, SOL_SOCKET, SO_RCVBUF, &rcvbuf, sizeof(rcvbuf));

    sockaddr_in local{};
    local.sin_family = AF_INET;
    local.sin_port = htons(ip.portDestination);
    local.sin_addr.s_addr = htonl(ip.ipDestination); // host->network

    if (::bind(rt.fd, reinterpret_cast<sockaddr *>(&local), sizeof(local)) < 0) {
      CALL_IF(AS_STRING("Failed to bind to "
                        << ipInt2Str(ip.ipDestination) << ":" << ip.portDestination
                        << " <-- " << ipInt2Str(ip.ipSource) << ":" << ip.portSource));
      success = false;
      continue;
    }

    rt.base = event_base_new();
    if (!rt.base) {
      CALL_IF("Failed to create event_base.");
      success = false;
      continue;
    }

    CALL_IF(AS_STRING("Successfully bind to "
                      << ipInt2Str(ip.ipDestination) << ":" << ip.portDestination
                      << " <-- " << ipInt2Str(ip.ipSource) << ":" << ip.portSource));
  }

  if (!success)
    throw openpni::exceptions::resource_unavailable();

  return success;
}

void Socket::SocketImpl::threadSocket(int threadId) {
  auto &rt = m_runtime->m_socketRuntimes[threadId];
  const auto &ipSetting = mc_param.channelSettings[threadId];

  auto ctx = std::make_shared<Socket::SocketImpl::ThreadCtx>(
      this, threadId,
      ipSetting.ipDestination, // host-order
      ipSetting.portDestination, mc_param.storageUnitSize);

  rt.ev_read = event_new(rt.base, rt.fd, EV_READ | EV_PERSIST, &on_udp_read, ctx.get());
  if (!rt.ev_read) {
    CALL_IF("event_new(read) failed");
    return;
  }
  event_add(rt.ev_read, nullptr);

  timeval tv{};
  tv.tv_sec = 0;
  tv.tv_usec = 1000; // 1ms
  rt.ev_tick = event_new(rt.base, -1, EV_TIMEOUT | EV_PERSIST, &on_tick, ctx.get());
  if (!rt.ev_tick) {
    CALL_IF("event_new(tick) failed");
    event_free(rt.ev_read);
    return;
  }
  event_add(rt.ev_tick, &tv);

  event_base_dispatch(rt.base);

  ctx->flush_if_needed(true);

  if (rt.ev_tick) {
    event_free(rt.ev_tick);
    rt.ev_tick = nullptr;
  }
  if (rt.ev_read) {
    event_free(rt.ev_read);
    rt.ev_read = nullptr;
  }
  if (rt.base) {
    event_base_free(rt.base);
    rt.base = nullptr;
  }
  if (rt.fd >= 0) {
    evutil_closesocket(rt.fd);
    rt.fd = -1;
  }
}

void Socket::SocketImpl::threadMark() {
  auto startTime = cro::steady_clock::now();
  uint64_t sleepTime = 0;
  uint64_t lastMarkClock = 0;
  while (!m_runtime->m_stopped) {
    std::this_thread::sleep_until(
        startTime + cro::milliseconds(++sleepTime * mc_param.timeSwitchBuffer_ms));
#ifdef PNI_USE_CAS_LOOP
    uint64_t writingIndex = m_buffer->writingFinishIndex;
#else
    uint64_t writingIndex = m_buffer->writingIndex;
#endif
    const auto markClock = std::chrono::duration_cast<std::chrono::milliseconds>(
                               std::chrono::steady_clock::now() - m_buffer->clockStart)
                               .count();
    timed_buffer::mark(*m_buffer, writingIndex, markClock, markClock - lastMarkClock);
    lastMarkClock = markClock;
  }
}
} // namespace openpni::process::socket
