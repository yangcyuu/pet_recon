#pragma once
#include <vector>
namespace openpni::common {
class ReuseableStream {
public:
  ReuseableStream() = default;
  ReuseableStream(const ReuseableStream &) = delete;
  ReuseableStream(ReuseableStream &&) = default;
  ReuseableStream &operator=(const ReuseableStream &) = delete;
  ReuseableStream &operator=(ReuseableStream &&) = default;

public:
  void clear() {
    m_buffer.clear();
    m_writePos = 0;
  }
  void reuse() { m_writePos = 0; }
  void setPos(std::size_t pos) {
    if (pos > m_buffer.size())
      m_buffer.resize(pos);
    m_writePos = pos;
  }
  std::size_t pos() const { return m_writePos; }
  bool empty() const { return m_writePos == 0; }
  void write(const char *data, std::size_t size) {
    if (m_writePos + size > m_buffer.size()) {
      m_buffer.resize(m_writePos + size);
    }
    std::copy(data, data + size, m_buffer.data() + m_writePos);
    m_writePos += size;
  }
  void write(const char *begin, const char *end) { write(begin, end - begin); }
  const char *data() const { return m_buffer.data(); }
  const char *begin() const { return m_buffer.data(); }
  const char *end() const { return m_buffer.data() + m_writePos; }

private:
  std::vector<char> m_buffer; // 内部缓冲区
  std::size_t m_writePos{0};  // 读取位置
};
} // namespace openpni::common
