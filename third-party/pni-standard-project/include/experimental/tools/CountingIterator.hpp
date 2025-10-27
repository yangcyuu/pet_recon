#pragma once
#include <utility>
namespace openpni::experimental::tools {
template <typename Parent, typename Func>
struct CountingIterator {
  const Parent &parent;
  int64_t count = 0;
  Func func;

  using value_type = decltype(func(std::declval<Parent>(), 0));
  using difference_type = std::ptrdiff_t;
  using pointer = value_type *;
  using reference = value_type &;
  //   using iterator_category = std::input_iterator_tag;
  //   using iterator_concept = std::input_iterator_tag;
  __PNI_CUDA_MACRO__ CountingIterator(
      const Parent &p, Func f, int64_t count = 0)
      : parent(p)
      , func(f)
      , count(count) {}
  __PNI_CUDA_MACRO__ CountingIterator(
      const CountingIterator &other)
      : parent(other.parent)
      , func(other.func)
      , count(other.count) {}
  __PNI_CUDA_MACRO__ CountingIterator &operator=(
      const CountingIterator &other) {
    if (this != &other) {
      // parent = other.parent; // parent is const reference, do not assign
      func = other.func;
    }
    return *this;
  }
  __PNI_CUDA_MACRO__ CountingIterator &operator++() {
    count++;
    return *this;
  }
  __PNI_CUDA_MACRO__ CountingIterator operator++(
      int) {
    CountingIterator tmp = *this;
    ++(*this);
    return tmp;
  }
  __PNI_CUDA_MACRO__ bool operator==(
      const CountingIterator &other) const {
    return count == other.count;
  }
  __PNI_CUDA_MACRO__ bool operator!=(
      const CountingIterator &other) const {
    return !(*this == other);
  }
  __PNI_CUDA_MACRO__ value_type operator*() const { return func(parent, count); }
};
} // namespace  openpni::experimental::tools
