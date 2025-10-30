#pragma once
#ifndef _PNI_STD_TIMED_DATA_BUFFER_HPP_
#define _PNI_STD_TIMED_DATA_BUFFER_HPP_
#include <atomic>
#include <chrono>
#include <cstring>
#include <iostream>
#include <memory>
#include <optional>
#include <shared_mutex>
#include <thread>

#include "include/misc/CycledBuffer.hpp"
#include "include/process/Acquisition.hpp"
// #define PNI_USE_CAS_LOOP
namespace openpni::process::timed_buffer {
constexpr uint64_t read_mark_buffer_size = 4096;
struct ReadMark {
  uint64_t indexEnd;
  uint64_t clockEnd;
  uint32_t duration;
};
struct Groups {
  std::unique_ptr<uint8_t[]> data;
  std::unique_ptr<uint16_t[]> length;
  std::unique_ptr<uint64_t[]> offset;
  std::unique_ptr<uint16_t[]> channel;
  std::unique_ptr<int64_t[]> clock;
  std::atomic_flag writeThreadGo;
  std::atomic<uint64_t> writingIndex{0};
  uint64_t markingIndex{0};
  std::atomic<uint64_t> empty;
  uint64_t pauseThreshold;
  uint64_t realAllocatePacketNum;
  uint16_t maxPacketSize;
  std::unique_ptr<common::CycledBuffer<ReadMark, false>> readBuffer;
  std::chrono::time_point<std::chrono::steady_clock> clockStart;
  std::atomic<uint64_t> totalRxBytes;
#ifdef PNI_USE_CAS_LOOP
  std::atomic<uint64_t> writingFinishIndex{0};
#else
  std::shared_mutex smutex;
#endif // PNI_USE_CAS_LOOP
};
template <typename T>
inline void new_(
    std::unique_ptr<T[]> &p, size_t l) {
  p.reset(new T[l]);
  ::memset(&p[0], 0, sizeof(T) * l);
}
template <typename T>
inline void new__(
    std::unique_ptr<T[]> &p, size_t l) {
  p.reset(new T[l]);
  const uint64_t byteSize = sizeof(T) * l;
  const auto threadNum =
      std::max<uint64_t>(1, std::min<uint64_t>(byteSize >> 31, std::thread::hardware_concurrency() / 2));
  const auto singleThreadBytes = byteSize / threadNum;
  std::vector<std::jthread> vecThreads;
  for (int i = 0; i < threadNum; i++)
    vecThreads.emplace_back([&p, byte_start = i * singleThreadBytes,
                             byte_end = i == threadNum - 1 ? byteSize : (i + 1) * singleThreadBytes] noexcept {
      ::memset(((char *)p.get()) + byte_start, 0, byte_end - byte_start);
    });
}
inline std::unique_ptr<Groups> makeGroups(
    uint64_t __bufferPacketNum, uint16_t __maxPacketSize, uint16_t __maxBurstSize, uint16_t __maxThreads) {
  std::unique_ptr<Groups> result;
  try {
    result = std::make_unique<Groups>();
    result->pauseThreshold = 2 * __maxBurstSize * (__maxThreads + 1);
    result->realAllocatePacketNum = __bufferPacketNum + result->pauseThreshold;
    new_(result->channel, result->realAllocatePacketNum);
    new__(result->data, result->realAllocatePacketNum * __maxPacketSize);
    new_(result->length, result->realAllocatePacketNum);
    new_(result->offset, result->realAllocatePacketNum);
    new_(result->clock, result->realAllocatePacketNum);
    result->maxPacketSize = __maxPacketSize;
    result->empty = result->realAllocatePacketNum;
    result->writeThreadGo.test_and_set();
    result->readBuffer = std::make_unique<decltype(result->readBuffer)::element_type>(read_mark_buffer_size);
  } catch (const std::exception &e) {
    return decltype(result)();
  }
  return result;
}
inline void clear(
    Groups &__group) {
  __group.empty = __group.realAllocatePacketNum;
  __group.markingIndex = 0;
  __group.readBuffer = std::make_unique<decltype(__group.readBuffer)::element_type>(read_mark_buffer_size);
  __group.totalRxBytes = 0;
  __group.writeThreadGo.test_and_set();
  __group.writingIndex = 0;
}
inline void start(
    Groups &__group) {
  __group.clockStart = std::chrono::steady_clock::now();
}
inline bool write(
    Groups &__group, const uint8_t *const __src, const uint16_t *const __length, const uint16_t *const __channel,
    const uint32_t *const __offset, uint16_t __burst) noexcept {
  if (!__burst)
    return true;
  if (!__group.writeThreadGo.test())
    return false;

#ifndef PNI_USE_CAS_LOOP
  std::shared_lock lll(__group.smutex);
#endif // PNI_USE_CAS_LOOP
  const auto empty = __group.empty.fetch_sub(__burst) - __burst;
  if (empty <= __group.pauseThreshold)
    __group.writeThreadGo.clear();
  auto _currentWritingIndex = __group.writingIndex.fetch_add(__burst);
  const auto currentWritingIndex = _currentWritingIndex % __group.realAllocatePacketNum;

  const uint32_t clockNow =
      std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - __group.clockStart)
          .count();
  if (currentWritingIndex + __burst > __group.realAllocatePacketNum) {
    const auto burst1 = __group.realAllocatePacketNum - currentWritingIndex;
    const auto burst2 = __burst - burst1;
    for (uint16_t i = 0; i < burst1; i++) {
      __group.channel[currentWritingIndex + i] = __channel[i];
      __group.length[currentWritingIndex + i] = __length[i];
      __group.offset[currentWritingIndex + i] = currentWritingIndex * __group.maxPacketSize + __offset[i] - __offset[0];
      __group.totalRxBytes += __length[i];
      __group.clock[currentWritingIndex + i] = clockNow;
    }
    for (uint16_t i = 0; i < burst2; i++) {
      __group.channel[i] = __channel[i + burst1];
      __group.length[i] = __length[i + burst1];
      __group.offset[i] = __offset[i + burst1] - __offset[burst1];
      __group.totalRxBytes += __length[i + burst1];
      __group.clock[i] = clockNow;
    }
    memcpy(&__group.data[__group.maxPacketSize * currentWritingIndex], &__src[__offset[0]],
           __offset[burst1 - 1] + __length[burst1 - 1] - __offset[0]);
    memcpy(&__group.data[0], &__src[__offset[burst1]],
           __offset[__burst - 1] + __length[__burst - 1] - __offset[burst1]);
  } else {
    for (uint16_t i = 0; i < __burst; i++) {
      __group.channel[currentWritingIndex + i] = __channel[i];
      __group.length[currentWritingIndex + i] = __length[i];
      __group.offset[currentWritingIndex + i] = currentWritingIndex * __group.maxPacketSize + __offset[i] - __offset[0];
      __group.totalRxBytes += __length[i];
      __group.clock[currentWritingIndex + i] = clockNow;
    }
    memcpy(&__group.data[__group.maxPacketSize * currentWritingIndex], &__src[__offset[0]],
           __offset[__burst - 1] + __length[__burst - 1] - __offset[0]);
  }

#ifdef PNI_USE_CAS_LOOP
  while (true) {
    auto _currentWritingIndex_ = _currentWritingIndex;
    if (__group.writingFinishIndex.compare_exchange_strong(_currentWritingIndex_, _currentWritingIndex + __burst))
      break;
    ;
  }
#endif // PNI_USE_CAS_LOOP
  return true;
}

inline std::optional<ReadMark> read(
    Groups &__group) noexcept {
  std::optional<ReadMark> result;
  __group.readBuffer->read(
      [&result](const decltype(__group.readBuffer)::element_type::BufferType &buffer) noexcept { result = buffer; });
  return result;
}
inline void mark(
    Groups &__group, uint64_t writingIndex, uint64_t clockEnd, uint32_t duration) noexcept {
  __group.readBuffer->write(
      [&writingIndex, &clockEnd, &duration](decltype(__group.readBuffer)::element_type::BufferType &buffer) noexcept {
        buffer.indexEnd = writingIndex;
        buffer.clockEnd = clockEnd;
        buffer.duration = duration;
      });
}
inline void stop(
    Groups &__group) noexcept {
  __group.readBuffer->stop();
}
}; // namespace openpni::process::timed_buffer

#endif // _PNI_STD_GOURPED_DATA_BUFFER_HPP_