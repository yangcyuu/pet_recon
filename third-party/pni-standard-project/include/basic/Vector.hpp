#pragma once
#include "Math.hpp"
namespace openpni::basic {
template <typename T, int N>
  requires(N >= 2)
struct Vector {
  T data[N]{};
  __PNI_CUDA_MACRO__ static constexpr Vector create() { return Vector(); }
  __PNI_CUDA_MACRO__ static constexpr Vector create(
      T v) {
    Vector result;
    for (int i = 0; i < N; i++)
      result.data[i] = v;
    return result;
  }
  template <typename... Args>
  __PNI_CUDA_MACRO__ static constexpr Vector create(
      Args... args) {
    static_assert(sizeof...(Args) == N, "Number of arguments must match vector dimension");
    return Vector{static_cast<T>(args)...};
  }
  __PNI_CUDA_MACRO__ constexpr Vector<T, N + 1> expand(
      T v) const {
    Vector<T, N + 1> result;
    *reinterpret_cast<Vector<T, N> *>(&result) = *this;
    result.data[N] = v;
    return result;
  }

  __PNI_CUDA_MACRO__ constexpr T &operator[](
      int index) {
    return data[index];
  }
  __PNI_CUDA_MACRO__ constexpr const T &operator[](
      int index) const {
    return data[index];
  }
  __PNI_CUDA_MACRO__ constexpr int size() const { return N; }
  template <typename TT = T>
  __PNI_CUDA_MACRO__ auto operator+(
      const Vector<TT, N> &other) const -> Vector<decltype(TT() + T()), N> {
    Vector<decltype(TT() + T()), N> result;
    for (int i = 0; i < N; i++)
      result[i] = data[i] + other[i];
    return result;
  }
  template <typename TT = T>
  __PNI_CUDA_MACRO__ auto operator-(
      const Vector<TT, N> &other) const -> Vector<decltype(TT() - T()), N> {
    Vector<decltype(TT() - T()), N> result;
    for (int i = 0; i < N; i++)
      result[i] = data[i] - other[i];
    return result;
  }
  template <typename TT = T>
  __PNI_CUDA_MACRO__ auto operator*(
      const Vector<TT, N> &other) const -> Vector<decltype(TT() * T()), N> {
    Vector<decltype(TT() * T()), N> result;
    for (int i = 0; i < N; i++)
      result[i] = data[i] * other[i];
    return result;
  }
  template <typename TT = T>
  __PNI_CUDA_MACRO__ auto operator/(
      const Vector<TT, N> &other) const -> Vector<decltype(TT() / T()), N> {
    Vector<decltype(TT() / T()), N> result;
    for (int i = 0; i < N; i++)
      result[i] = data[i] / other[i];
    return result;
  }
  template <typename TT = T>
  __PNI_CUDA_MACRO__ auto dot(
      const Vector<TT, N> &other) const -> decltype(TT() * T()) {
    decltype(TT() * T()) result = 0;
    for (int i = 0; i < N; i++)
      result += data[i] * other[i];
    return result;
  }
  __PNI_CUDA_MACRO__ T l1() const {
    T result = 0;
    for (int i = 0; i < N; i++)
      result += data[i];
    return result;
  }
  __PNI_CUDA_MACRO__ T l22() const {
    T result = 0;
    for (int i = 0; i < N; i++)
      result += data[i] * data[i];
    return result;
  }

  template <FloatingPoint_c F>
  __PNI_CUDA_MACRO__ F l2() const {
    return basic::FMath<F>(static_cast<F>(l22()));
  }

  T min() const {
    T result = data[0];
    for (int i = 1; i < N; i++)
      if (data[i] < result)
        result = data[i];
    return result;
  }
  T max() const {
    T result = data[0];
    for (int i = 1; i < N; i++)
      if (data[i] > result)
        result = data[i];
    return result;
  }
};

template <typename Parent, typename Func>
struct CountingIterator {
  const Parent &parent;
  int64_t count = 0;
  Func func;

  using value_type = decltype(func(std::declval<Parent>(), 0));
  using difference_type = std::ptrdiff_t;
  using pointer = value_type *;
  using reference = value_type &;
  using iterator_category = std::input_iterator_tag;
  using iterator_concept = std::input_iterator_tag;
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
template <int N>
struct MDSpan {
  Vector<int64_t, N> dimSize;
  int64_t total_size;

  template <typename... Args>
  __PNI_CUDA_MACRO__ static constexpr auto create(
      Args... __extents) {
    static_assert(sizeof...(__extents) == N, "Number of arguments must match vector dimension");
    static_assert((std::is_integral_v<Args> && ...), "All arguments must be integral");
    int64_t values[] = {static_cast<int64_t>(__extents)...};
    MDSpan result;
    result.total_size = 1;
    for (int i = 0; i < N; i++) {
      result.dimSize[i] = values[i];
      result.total_size *= values[i];
    }
    return result;
  }

  constexpr int64_t totalSize() const { return total_size; }

  __PNI_CUDA_MACRO__ auto operator()(
      const Vector<int64_t, N> &__index) const {
    int64_t result = __index[N - 1];
    for (int i = N - 2; i >= 0; i--)
      result = result * dimSize[i] + __index[i];
    return result;
  }
  __PNI_CUDA_MACRO__ auto operator[](
      const Vector<int64_t, N> &__index) const {
    return (*this)(__index);
  }
  template <typename... Args>
  __PNI_CUDA_MACRO__ auto operator()(
      Args... __extents) const {
    return (*this)(Vector<int64_t, N>::create(__extents...));
  }

  static Vector<int64_t, N + 1> createIndex(
      const MDSpan &_this, int64_t index) {
    Vector<int64_t, N + 1> result = Vector<int64_t, N + 1>::create();
    result[N] = index;
    for (int i = 0; i < N; i++) {
      result[i] = index % _this.dimSize[i];
      index /= _this.dimSize[i];
    }
    return result;
  }

  using Iterator = CountingIterator<MDSpan, decltype(&MDSpan::createIndex)>;

  __PNI_CUDA_MACRO__ Iterator begin() const { return Iterator(*this, &MDSpan::createIndex, 0); }
  __PNI_CUDA_MACRO__ Iterator end() const {
    int64_t end_fake = total_size;
    return Iterator(*this, &MDSpan::createIndex, end_fake);
  }
};
template <int N>
struct MDBeginEndSpan {
  Vector<int64_t, N> begins;
  Vector<int64_t, N> ends;
  int64_t total_size;
  __PNI_CUDA_MACRO__ static constexpr auto create(
      Vector<int64_t, N> __begin, Vector<int64_t, N> __end) {
    MDBeginEndSpan result;
    result.begins = __begin;
    result.ends = __end;
    result.total_size = 1;
    for (int i = 0; i < N; i++)
      result.total_size *= (__end[i] - __begin[i]);
    return result;
  }
  __PNI_CUDA_MACRO__ constexpr int64_t totalSize() const { return total_size; }

  static Vector<int64_t, N + 1> createIndex(
      const MDBeginEndSpan &_this, int64_t index) {
    Vector<int64_t, N + 1> result = Vector<int64_t, N + 1>::create();
    result[N] = index;
    auto dimSize = _this.ends - _this.begins;
    for (int i = 0; i < N; i++) {
      result[i] = index % dimSize[i];
      index /= dimSize[i];
    }
    result = result + _this.begins;
    return result;
  }

  using Iterator = CountingIterator<MDBeginEndSpan, decltype(&MDBeginEndSpan::createIndex)>;
  __PNI_CUDA_MACRO__ Iterator begin() const { return Iterator(*this, &MDBeginEndSpan::createIndex, 0); }
  __PNI_CUDA_MACRO__ Iterator end() const {
    int64_t end_fake = total_size;
    return Iterator(*this, &MDBeginEndSpan::createIndex, end_fake);
  }
};

} // namespace openpni::basic
namespace openpni::_vec {
using p3df = openpni::basic::Vector<float, 3>;
using p2df = openpni::basic::Vector<float, 2>;
using p3di = openpni::basic::Vector<int, 3>;
using p2di = openpni::basic::Vector<int, 2>;
using p3dd = openpni::basic::Vector<double, 3>;
using p2dd = openpni::basic::Vector<double, 2>;
template <typename T>
using Vec2 = openpni::basic::Vector<T, 2>;
template <typename T>
using Vec3 = openpni::basic::Vector<T, 3>;
using Span3 = openpni::basic::MDSpan<3>;
using Span2 = openpni::basic::MDSpan<2>;
} // namespace openpni::_vec
