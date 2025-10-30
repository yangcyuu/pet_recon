#pragma once
#include <functional>
#include <memory>

#include "../basic/PetDataType.h"
#include "../io/IO.hpp"
namespace openpni::misc {
class ListmodeBuffer_impl;
class ListmodeBuffer {
public:
  using FlushFunction = std::function<void(const basic::Listmode_t *, std::size_t)>;

public:
  ListmodeBuffer();
  virtual ~ListmodeBuffer();

public:
  ListmodeBuffer &setBufferSize(std::size_t __count);
  ListmodeBuffer &callWhenFlush(FlushFunction &&__func);
  ListmodeBuffer &append(basic::Listmode_t *__data, std::size_t __count);
  ListmodeBuffer &append(io::ListmodeFileInput &__file,
                         std::vector<io::_SegmentView> __segmentViews);
  ListmodeBuffer &flush();

private:
  std::unique_ptr<ListmodeBuffer_impl> m_impl;
};

class ListmodeBufferAsync_impl;
class ListmodeBufferAsync {
public:
  using FlushFunction = std::function<void(const basic::Listmode_t *, std::size_t)>;

public:
  ListmodeBufferAsync();
  virtual ~ListmodeBufferAsync();

public:
  ListmodeBufferAsync &setBufferSize(std::size_t __count);
  ListmodeBufferAsync &callWhenFlush(FlushFunction &&__func);
  ListmodeBufferAsync &append(basic::Listmode_t *__data, std::size_t __count);
  ListmodeBufferAsync &append(io::ListmodeFileInput &__file,
                              std::vector<io::_SegmentView> __segmentViews);
  ListmodeBufferAsync &flush();

private:
  std::unique_ptr<ListmodeBufferAsync_impl> m_impl;
};
} // namespace openpni::misc
