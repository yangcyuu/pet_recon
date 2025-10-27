#pragma once
#include <atomic>
#include <functional>
#include <memory>
#include <thread>
namespace openpni::common {
// 仅限单生产者单消费者使用
template <typename _BufferType>
class AtomicRingBuffer {
public:
  AtomicRingBuffer(std::size_t size)
      : m_size(size), m_buffer(std::make_unique<BufferItem[]>(size)) {}

  using BufferType = _BufferType;

private:
  struct BufferItem {
    BufferType buffer;
    bool EOB{false}; // End of buffer
  };

public:
  void write(const std::function<void(BufferType &)> &&__funcWrite) noexcept {
    auto writeIndex = m_writeIndex.load();
    while (writeIndex >= m_readIndex.load() + m_size) {
      // Wait until there is space in the buffer
      std::this_thread::sleep_for(std::chrono::microseconds(1));
    }
    __funcWrite(m_buffer[writeIndex % m_size].buffer);
    m_buffer[writeIndex % m_size].EOB = false; // Mark as not end of buffer
    m_writeIndex.fetch_add(1);
  };
  bool read(const std::function<void(const BufferType &)> &&__funcRead) noexcept {
    auto readIndex = m_readIndex.load();
    while (readIndex >= m_writeIndex.load()) {
      // Wait until there is data in the buffer
      std::this_thread::sleep_for(std::chrono::microseconds(1));
    }
    auto &item = m_buffer[readIndex % m_size];
    if (item.EOB) {
      stop();
      m_readIndex++;
      return false; // End of buffer reached
    }
    __funcRead(item.buffer);
    m_readIndex++;
    return true;
  };
  void stop() noexcept {
    auto writeIndex = m_writeIndex.load();
    while (writeIndex >= m_readIndex.load() + m_size) {
      // Wait until there is space in the buffer
      std::this_thread::sleep_for(std::chrono::microseconds(1));
    }
    m_buffer[writeIndex % m_size].EOB = true; // Mark as end of buffer
    m_writeIndex.fetch_add(1);
  }

private:
  std::unique_ptr<BufferItem[]> m_buffer;
  std::size_t m_size{0};
  std::atomic<uint64_t> m_writeIndex{0};
  std::atomic<uint64_t> m_readIndex{0};
};
} // namespace openpni::common
