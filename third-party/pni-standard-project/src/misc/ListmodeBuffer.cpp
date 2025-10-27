#include "include/misc/ListmodeBuffer.hpp"

#include <cstring>
#include <future>

#include "include/io/Decoding.hpp"
namespace openpni::misc {
template <typename T>
void appendToBuffer(
    T &__buffer, io::ListmodeFileInput &__file, std::vector<io::_SegmentView> __segmentViews) {
  std::vector<basic::Listmode_t> listmodes;
  for (std::size_t i = 0; i < __segmentViews.size(); ++i) {
    const auto &view = __segmentViews[i];
    const auto segmentHeader = __file.segmentHeader(view.segmentIndex);
    const auto segmentData = __file.readSegment(
        view.segmentIndex, i == __segmentViews.size() - 1 ? uint32_t(-1) : __segmentViews[i + 1].segmentIndex);
    if (listmodes.size() < segmentHeader.count)
      listmodes.resize(segmentHeader.count);
    io::listmode::decompress(listmodes.data(), __file.header(), segmentHeader, segmentData);
    __buffer.append(listmodes.data()+view.dataIndexBegin, view.dataIndexEnd-view.dataIndexBegin);
  }
}

using FlushFunction = ListmodeBuffer::FlushFunction;
class ListmodeBuffer_impl {
public:
  ListmodeBuffer_impl() = default;
  ~ListmodeBuffer_impl() { flush(); }

  void setBufferSize(
      std::size_t __count) {
    m_buffer = std::make_unique_for_overwrite<basic::Listmode_t[]>(__count);
    m_bufferSize = __count;
  }

  void callWhenFlush(
      FlushFunction &&__func) {
    m_flushFunction = std::move(__func);
  }

  void append(
      basic::Listmode_t *__data, std::size_t __count) {
    if (__count + m_currentSize > m_bufferSize) {
      memcpy(&m_buffer[m_currentSize], __data, (m_bufferSize - m_currentSize) * sizeof(basic::Listmode_t));
      __data += (m_bufferSize - m_currentSize);
      __count -= (m_bufferSize - m_currentSize);
      flush();
      append(__data, __count);
    } else {
      memcpy(&m_buffer[m_currentSize], __data, __count * sizeof(basic::Listmode_t));
      m_currentSize += __count;
    }
  }
  void append(
      io::ListmodeFileInput &__file, std::vector<io::_SegmentView> __segmentViews) {
    appendToBuffer(*this, __file, __segmentViews);
  }

  void flush() {
    if (m_currentSize == 0 || !m_flushFunction)
      return;
    if (m_flushFunction)
      m_flushFunction(&m_buffer[0], m_currentSize);
    m_currentSize = 0;
  }

private:
  std::unique_ptr<basic::Listmode_t[]> m_buffer;
  std::size_t m_bufferSize;
  std::size_t m_currentSize = 0;
  FlushFunction m_flushFunction;
};
ListmodeBuffer &ListmodeBuffer::setBufferSize(
    std::size_t __count) {
  m_impl->setBufferSize(__count);
  return *this;
}
ListmodeBuffer &ListmodeBuffer::callWhenFlush(
    FlushFunction &&__func) {
  m_impl->callWhenFlush(std::move(__func));
  return *this;
}

ListmodeBuffer &ListmodeBuffer::append(
    basic::Listmode_t *__data, std::size_t __count) {
  m_impl->append(__data, __count);
  return *this;
}
ListmodeBuffer &ListmodeBuffer::append(
    io::ListmodeFileInput &__file, std::vector<io::_SegmentView> __segmentViews) {
  m_impl->append(__file, __segmentViews);
  return *this;
}
ListmodeBuffer &ListmodeBuffer::flush() {
  m_impl->flush();
  return *this;
}
ListmodeBuffer::~ListmodeBuffer() {}
ListmodeBuffer::ListmodeBuffer()
    : m_impl(std::make_unique<ListmodeBuffer_impl>()) {}

class ListmodeBufferAsync_impl {
public:
  using FlushFunction = std::function<void(const basic::Listmode_t *, std::size_t)>;

public:
  ListmodeBufferAsync_impl() {}
  virtual ~ListmodeBufferAsync_impl() {
    flush();
    waitForFlush();
  }

public:
  void setBufferSize(
      std::size_t __count) {
    m_bufferSize = __count;
    m_buffers[0] = std::make_unique_for_overwrite<basic::Listmode_t[]>(m_bufferSize);
    m_buffers[1] = std::make_unique_for_overwrite<basic::Listmode_t[]>(m_bufferSize);
  }
  void callWhenFlush(
      FlushFunction &&__func) {
    m_flushFunction = std::move(__func);
  }
  void append(
      basic::Listmode_t *__data, std::size_t __count) {
    if (__count + m_bufferInfo[m_activeBufferIndex].size > m_bufferSize) {
      memcpy(&m_buffers[m_activeBufferIndex][m_bufferInfo[m_activeBufferIndex].size], __data,
             (m_bufferSize - m_bufferInfo[m_activeBufferIndex].size) * sizeof(basic::Listmode_t));
      __data += (m_bufferSize - m_bufferInfo[m_activeBufferIndex].size);
      __count -= (m_bufferSize - m_bufferInfo[m_activeBufferIndex].size);
      m_bufferInfo[m_activeBufferIndex].size = m_bufferSize;
      flush();
      append(__data, __count);
    } else {
      memcpy(&m_buffers[m_activeBufferIndex][m_bufferInfo[m_activeBufferIndex].size], __data,
             __count * sizeof(basic::Listmode_t));
      m_bufferInfo[m_activeBufferIndex].size += __count;
    }
  }
  void append(
      io::ListmodeFileInput &__file, std::vector<io::_SegmentView> __segmentViews) {
    appendToBuffer(*this, __file, __segmentViews);
  }
  void flush() {
    if (m_flushFunction) {
      if (m_flushFuture.valid())
        m_bufferInfo[m_flushFuture.get()].readyToFlush = true;
      if (m_bufferInfo[m_activeBufferIndex].size != 0 && m_bufferInfo[m_activeBufferIndex].readyToFlush) {
        m_flushFuture = std::async(std::launch::async, [this, activeBufferIndex = m_activeBufferIndex]() -> int {
          m_flushFunction(m_buffers[activeBufferIndex].get(), m_bufferInfo[activeBufferIndex].size);
          m_bufferInfo[activeBufferIndex].size = 0;
          return activeBufferIndex;
        });
        m_bufferInfo[m_activeBufferIndex].readyToFlush = false;
      }
    }
    m_activeBufferIndex = (m_activeBufferIndex + 1) % 2;
  }
  void waitForFlush() {
    if (m_flushFuture.valid())
      m_bufferInfo[m_flushFuture.get()].readyToFlush = true;
  }

private:
  struct BufferInfo {
    std::size_t size = 0;
    bool readyToFlush = true;
  };

private:
  std::array<std::unique_ptr<basic::Listmode_t[]>, 2> m_buffers;
  std::size_t m_bufferSize;
  std::array<BufferInfo, 2> m_bufferInfo;
  int m_activeBufferIndex = 0;
  FlushFunction m_flushFunction;
  std::future<int> m_flushFuture;
};

ListmodeBufferAsync &ListmodeBufferAsync::setBufferSize(
    std::size_t __count) {
  m_impl->setBufferSize(__count);
  return *this;
}
ListmodeBufferAsync &ListmodeBufferAsync::callWhenFlush(
    FlushFunction &&__func) {
  m_impl->callWhenFlush(std::move(__func));
  return *this;
}

ListmodeBufferAsync &ListmodeBufferAsync::append(
    basic::Listmode_t *__data, std::size_t __count) {
  m_impl->append(__data, __count);
  return *this;
}
ListmodeBufferAsync &ListmodeBufferAsync::append(
    io::ListmodeFileInput &__file, std::vector<io::_SegmentView> __segmentViews) {
  m_impl->append(__file, __segmentViews);
  return *this;
}
ListmodeBufferAsync &ListmodeBufferAsync::flush() {
  m_impl->flush();
  m_impl->waitForFlush();
  return *this;
}
ListmodeBufferAsync::~ListmodeBufferAsync() {}
ListmodeBufferAsync::ListmodeBufferAsync()
    : m_impl(std::make_unique<ListmodeBufferAsync_impl>()) {}
} // namespace openpni::misc
