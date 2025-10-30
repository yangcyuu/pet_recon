#include "DPDKImpl.hpp"
#if PNI_STANDARD_CONFIG_ENABLE_DPDK
#include <algorithm>
#include <ranges>
#include <sstream>

#include "Consts.hpp"
#include "DPDKFunctions.hpp"
#include "src/common/TemplateFunctions.hpp"
#define CALL_IF(AAA)                                                                                                   \
  if (m_logger)                                                                                                        \
    m_logger(AAA);
#define AS_STRING(expression) ((std::stringstream() << expression).str())
static std::string ipInt2Str(
    uint32_t ip) noexcept {
  return std::to_string((ip >> 24) & 0xFF) + "." + std::to_string((ip >> 16) & 0xFF) + "." +
         std::to_string((ip >> 8) & 0xFF) + "." + std::to_string(ip & 0xFF);
}
namespace c = openpni::consts;
namespace openpni::process::dpdk {
DPDK::DPDKImpl::DPDKImpl(
    AcquisitionInfo __param, process::FuncLogger &&__logger)
    : mc_param(__param)
    , m_logger(__logger)
    , m_ipMapper(toIPMapper(mc_param.channelSettings)) {

  CALL_IF(AS_STRING("Trying to allocate esitmated " << (double(mc_param.maxBufferSize) / c::GB) << " GB memory."))

  if (!allocate())
    m_status = false;
}

DPDK::DPDKImpl::~DPDKImpl() {}

Status DPDK::DPDKImpl::status() noexcept {
  Status result;
  result.totalRxBytes = m_buffer->totalRxBytes;
  result.totalRxPackets = m_buffer->writingIndex;
  result.volume = m_buffer->realAllocatePacketNum;
  result.used = result.volume - m_buffer->empty;
  result.unknown = m_runtime->r_unknownPkts;
  return result;
}

bool DPDK::DPDKImpl::start() {
  if (!m_status)
    return false;
  if (m_runtime) {
    timed_buffer::clear(*m_buffer);
  }
  m_runtime = std::make_unique<decltype(m_runtime)::element_type>();

  m_buffer->clockStart = std::chrono::steady_clock::now();
  m_funcWriteToMemory = [this](WriteParam *param) { writeToBuffer(param); };
  seizeDPDK(m_logger, m_funcWriteToMemory);
  m_runtime->r_dpdkSeized = true;
  m_runtime->r_threadMark = std::move(std::jthread(std::mem_fn(&DPDKImpl::threadMark), this));
  CALL_IF("Threads started.")
  return true;
}

bool DPDK::DPDKImpl::stop() noexcept {
  if (m_runtime && m_runtime->r_dpdkSeized)
    releaseDPDK();
  m_runtime->r_stopped = true;
  timed_buffer::stop(*m_buffer);
  return true;
}

bool DPDK::DPDKImpl::isFinished() noexcept {
  return m_runtime->r_finished;
}

std::optional<RawDataView> DPDK::DPDKImpl::read() noexcept {
  if (m_runtime->r_lastReadoutCount) {
    m_buffer->empty += m_runtime->r_lastReadoutCount;
    m_buffer->writeThreadGo.test_and_set();
    m_buffer->writeThreadGo.notify_all();
  }
  if (m_runtime->r_lastReadResult.has_value()) {
    decltype(m_runtime->r_lastReadResult) result;
    std::swap(result, m_runtime->r_lastReadResult);
    m_runtime->r_lastReadoutCount = result->count;
    return result;
  }
  auto readMark = timed_buffer::read(*m_buffer);
  if (readMark) {
#ifndef PNI_USE_CAS_LOOP
    std::lock_guard lll(m_buffer->smutex);
#endif // !PNI_USE_CAS_LOOP

    const auto indexEnd = readMark->indexEnd;
    RawDataView result;
    const auto [lastReadingRound, lastReadingIndexModded] =
        std::lldiv(m_runtime->r_lastReadingIndex, m_buffer->realAllocatePacketNum);
    const auto [thisReadingRound, thisReadingIndexModded] = std::lldiv(indexEnd, m_buffer->realAllocatePacketNum);
    if (thisReadingRound == lastReadingRound) { // 没有回转
      result.channel = &m_buffer->channel[lastReadingIndexModded];
      result.channelNum = mc_param.totalChannelNum;
      result.clock_ms = m_buffer->clock[lastReadingIndexModded];
      result.count = (indexEnd - m_runtime->r_lastReadingIndex) % m_buffer->realAllocatePacketNum;
      if (result.count)
        result.duration_ms = m_buffer->clock[lastReadingIndexModded + result.count - 1] - result.clock_ms;
      else
        result.duration_ms = 0;
      result.data = &m_buffer->data[0];
      result.length = &m_buffer->length[lastReadingIndexModded];
      result.offset = &m_buffer->offset[lastReadingIndexModded];
    } else { // 回转了
      result.channel = &m_buffer->channel[lastReadingIndexModded];
      result.channelNum = mc_param.totalChannelNum;
      result.clock_ms = m_buffer->clock[lastReadingIndexModded];
      result.count = m_buffer->realAllocatePacketNum - lastReadingIndexModded;
      if (result.count)
        result.duration_ms = m_buffer->clock[m_buffer->realAllocatePacketNum - 1] - result.clock_ms;
      else
        result.duration_ms = 0;
      result.data = &m_buffer->data[0];
      result.length = &m_buffer->length[lastReadingIndexModded];
      result.offset = &m_buffer->offset[lastReadingIndexModded];

      m_runtime->r_lastReadResult = RawDataView();
      m_runtime->r_lastReadResult->channel = &m_buffer->channel[0];
      m_runtime->r_lastReadResult->channelNum = mc_param.totalChannelNum;
      m_runtime->r_lastReadResult->count = indexEnd % m_buffer->realAllocatePacketNum;
      m_runtime->r_lastReadResult->clock_ms = m_buffer->clock[0];
      if (m_runtime->r_lastReadResult->count)
        m_runtime->r_lastReadResult->duration_ms =
            m_buffer->clock[m_runtime->r_lastReadResult->count - 1] - m_runtime->r_lastReadResult->clock_ms;
      else
        m_runtime->r_lastReadResult->duration_ms = 0;
      m_runtime->r_lastReadResult->data = &m_buffer->data[0];
      m_runtime->r_lastReadResult->length = &m_buffer->length[0];
      m_runtime->r_lastReadResult->offset = &m_buffer->offset[0];
    }
    m_runtime->r_lastReadingIndex = indexEnd;
    m_runtime->r_lastReadoutCount = result.count;
    return result;
  } else {
    m_runtime->r_finished = true;
    m_runtime->r_lastReadoutCount = 0;
    return std::nullopt;
  }
}

bool DPDK::DPDKImpl::allocate() {
  m_buffer = timed_buffer::makeGroups(mc_param.maxBufferSize / mc_param.storageUnitSize, mc_param.storageUnitSize,
                                      dpdk_move_burst, getDPDKMoveThreadNum());
  if (!m_buffer) {
    CALL_IF("Failed to allocate memory.")
    return false;
  }
  CALL_IF("Allocate finished.")
  return true;
}

void DPDK::DPDKImpl::removePacket(
    WriteParam *param) {
  uint16_t newBurst = 0;
  uint32_t newOffset = 0;
  uint32_t readOffset = 0;

  for (int i = 0; i < param->burst;) {
    // 跳过超限包
    if (param->length[i] > mc_param.storageUnitSize) {
      i++;
      continue;
    }

    // 找到一段连续合法包
    int j = i;
    while (j < param->burst && param->length[j] <= mc_param.storageUnitSize) {
      j++;
    }

    // 这段合法区间的总长度
    uint32_t segStart = param->offset[i];
    uint32_t segEnd = param->offset[j - 1] + param->length[j - 1];
    uint32_t segLen = segEnd - segStart;

    // 批量memmove
    if (segStart != newOffset) {
      memmove(param->src + newOffset, param->src + segStart, segLen);
    }

    // 更新包信息
    uint32_t dst = newOffset;
    for (int k = i; k < j; k++) {
      param->offset[newBurst] = dst;
      param->length[newBurst] = param->length[k];
      param->ipPair[newBurst] = param->ipPair[k];
      dst += param->length[k];
      newBurst++;
    }

    newOffset += segLen;
    i = j; // 继续下一段
  }
  param->burst = newBurst;

  // {
  //   uint16_t newBurst = 0;
  //   uint32_t newOffset = 0;
  //   for (int i = 0; i < param->burst; i++) {
  //     if (param->length[i] <= mc_param.storageUnitSize) {
  //       if (param->offset[i] != newOffset) {
  //         memmove(param->src + newOffset, param->src + param->offset[i],
  //                 param->length[i]);
  //       }
  //       param->offset[newBurst] = newOffset;
  //       param->length[newBurst] = param->length[i];
  //       param->ipPair[newBurst] = param->ipPair[i];
  //       newOffset += param->length[i];
  //       newBurst++;
  //     }
  //   }
  //   param->burst = newBurst;
  // }
}

bool DPDK::DPDKImpl::writeToBuffer(
    WriteParam *param) {
  if (param->burst == 0)
    return true;
  uint16_t channel[dpdk::dpdk_move_burst];
  const uint16_t storageUnitSize = mc_param.storageUnitSize;
  // 检查是否有超限包
  const bool hasOversize = std::ranges::any_of(std::span(param->length, param->burst),
                                               [storageUnitSize](uint16_t len) { return len > storageUnitSize; });

  if (hasOversize) {
    removePacket(param);
    if (param->burst == 0)
      return true;
  }

  for (int i = 0; i < param->burst; i++) {
    const uint16_t channelIndex = m_ipMapper(param->ipPair[i], param->src + param->offset[i], param->length[i]);
    if (channelIndex >= mc_param.totalChannelNum) {
      m_runtime->r_unknownPkts++;
      std::cerr << AS_STRING("Unknown channel index: " << channelIndex
                                                       << " for IP: " << ipInt2Str(param->ipPair[i].src_addr) << ":"
                                                       << param->ipPair[i].src_port << "\n");
    }
    channel[i] = channelIndex;
  }
  if (param->burst > 0) {
    if (!timed_buffer::write(*m_buffer, param->src, param->length, channel, param->offset, param->burst))
      m_buffer->writeThreadGo.wait(false);
  }
  return true;
}

void DPDK::DPDKImpl::threadMark() {
  namespace cro = std::chrono;
  auto startTime = cro::steady_clock::now();
  uint64_t sleepTime = 0;
  uint64_t lastMarkClock = 0;
  while (!m_runtime->r_stopped) {
    std::this_thread::sleep_until(startTime + cro::milliseconds(++sleepTime * mc_param.timeSwitchBuffer_ms));
#ifdef PNI_USE_CAS_LOOP
    uint64_t writingIndex = m_buffer->writingFinishIndex;
#else
    uint64_t writingIndex = m_buffer->writingIndex;
#endif // PNI_USE_CAS_LOOP

    const auto markClock =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - m_buffer->clockStart)
            .count();
    timed_buffer::mark(*m_buffer, writingIndex, markClock, markClock - lastMarkClock);
    lastMarkClock = markClock;
  }
}

} // namespace openpni::process::dpdk
#endif