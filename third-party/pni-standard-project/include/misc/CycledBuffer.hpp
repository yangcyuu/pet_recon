#pragma once
#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <semaphore>
#include <vector>
namespace openpni::common {
template <typename Semaphore, typename Time>
inline void acquireSemaphore(
    Semaphore &semaphore, const Time &timeOut) {
  while (!semaphore.try_acquire_for(timeOut))
    ;
}

template <typename _BufferType, int _AcquireTimeMicro = 0>
class CycledBuffer {
public:
  using BufferType = _BufferType;

private:
  struct BufferItem {
    BufferType buffer;
    bool EOB{false}; // End of buffer
    std::binary_semaphore lockEmpty{1};
    std::binary_semaphore lockData{0};
  };
  using BufferListType = std::unique_ptr<BufferItem[]>;

public:
  CycledBuffer(
      std::size_t BufferSize)
      : m_bufferSize(BufferSize)
      , m_list(std::make_unique<BufferItem[]>(BufferSize))
      , m_semaphoreEmpty(BufferSize) {}
  CycledBuffer(
      std::size_t BufferSize, std::function<void(BufferType &)> &&__funcInit)
      : m_bufferSize(BufferSize)
      , m_list(std::make_unique<BufferItem[]>(BufferSize))
      , m_semaphoreEmpty(BufferSize) {
    for (std::size_t i = 0; i < BufferSize; ++i)
      __funcInit(m_list[i].buffer);
  }
  ~CycledBuffer() {
    stop();
    clear();
  }

public:
  void stop() noexcept // 由于存在析构顺序的问题，建议stop()需要显式调用，否则可能导致线程卡住
  {
    std::call_once(m_stopped, [this]() noexcept { insertEOB(); });
  }
  bool read(
      std::function<void(const _BufferType &)> &&__funcRead) noexcept {
    if constexpr (_AcquireTimeMicro > 0)
      acquireSemaphore(m_semaphoreData, std::chrono::microseconds(_AcquireTimeMicro));
    else
      m_semaphoreData.acquire();
    const auto index = (m_indexNextRead++) % m_bufferSize;
    m_list[index].lockData.acquire();
    bool isEOB = m_list[index].EOB;
    if (!isEOB)
      __funcRead(m_list[index].buffer);
    m_list[index].lockEmpty.release();
    m_semaphoreEmpty.release();
    if (isEOB)
      insertEOB();
    return !isEOB;
  }
  void write(
      std::function<void(_BufferType &)> &&__funcWrite) noexcept {
    if constexpr (_AcquireTimeMicro > 0)
      acquireSemaphore(m_semaphoreEmpty, std::chrono::microseconds(_AcquireTimeMicro));
    else
      m_semaphoreEmpty.acquire();
    const auto index = (m_indexNextWrite++) % m_bufferSize;
    m_list[index].lockEmpty.acquire();
    __funcWrite(m_list[index].buffer);
    m_list[index].lockData.release();
    m_semaphoreData.release();
  }
  void clear() noexcept {
    while (read([](const _BufferType &) noexcept {}))
      ;
  }

private:
  void insertEOB() noexcept {
    m_semaphoreEmpty.acquire();
    const auto index = (m_indexNextWrite++) % m_bufferSize;
    m_list[index].lockEmpty.acquire();
    m_list[index].EOB = true;
    m_list[index].lockData.release();
    m_semaphoreData.release();
  }

private:
  BufferListType m_list;
  std::atomic<uint64_t> m_indexNextRead{0};
  std::atomic<uint64_t> m_indexNextWrite{0};
  std::counting_semaphore<> m_semaphoreData{0};
  std::counting_semaphore<> m_semaphoreEmpty;
  std::once_flag m_stopped;
  const std::size_t m_bufferSize;
};

template <typename _BufferType>
using NoDeadWaitCycledBuffer = CycledBuffer<_BufferType, 1>;
template <typename _BufferType>
using MultiCycledBuffer = CycledBuffer<_BufferType, 0>;

} // namespace openpni::common
