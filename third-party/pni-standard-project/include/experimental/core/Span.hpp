#pragma once
#include <cstdint>

#include "../tools/CountingIterator.hpp"
#include "Vector.hpp"

namespace openpni::experimental::core {
template <int N>
struct MDSpan {
  using index_type = Vector<int64_t, N>;
  constexpr static int dimension = N;

  Vector<int64_t, N> dimSize;

  __PNI_CUDA_MACRO__ static constexpr auto create() {
    MDSpan result;
    result.dimSize = Vector<int64_t, N>::create(1);
    return result;
  }

  __PNI_CUDA_MACRO__ static constexpr auto create(
      const Vector<int64_t, N> &__dimSize) {
    MDSpan result;
    result.dimSize = __dimSize;
    return result;
  }

  template <typename... Args>
    requires(sizeof...(Args) == N && (std::is_integral_v<Args> && ...))
  __PNI_CUDA_MACRO__ static constexpr auto create(
      Args... __extents) {
    static_assert(sizeof...(__extents) == N, "Number of arguments must match vector dimension");
    static_assert((std::is_integral_v<Args> && ...), "All arguments must be integral");
    int64_t values[] = {static_cast<int64_t>(__extents)...};
    MDSpan result;
    for (int i = 0; i < N; i++) {
      result.dimSize[i] = values[i];
    }
    return result;
  }

  __PNI_CUDA_MACRO__ constexpr int64_t totalSize() const {
    int64_t result = 1;
    for (int i = 0; i < N; i++)
      result *= dimSize[i];
    return result;
  }

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

  __PNI_CUDA_MACRO__ bool inBounds(
      const Vector<int64_t, N> &__index) const {
    for (int i = 0; i < N; i++)
      if (__index[i] < 0 || __index[i] >= dimSize[i])
        return false;
    return true;
  }
  template <typename... Args>
  __PNI_CUDA_MACRO__ bool inBounds(
      Args... __extents) const {
    return inBounds(Vector<int64_t, N>::create(__extents...));
  }

  __PNI_CUDA_MACRO__ static Vector<int64_t, N> createIndex(
      const MDSpan &_this, int64_t index) {
    Vector<int64_t, N> result = Vector<int64_t, N>::create();
    for (int i = 0; i < N; i++) {
      result[i] = index % _this.dimSize[i];
      index /= _this.dimSize[i];
    }
    return result;
  }

  __PNI_CUDA_MACRO__ Vector<int64_t, N> toIndex(
      int64_t index) const {
    Vector<int64_t, N> result = Vector<int64_t, N>::create();
    for (int i = 0; i < N; i++) {
      result[i] = index % dimSize[i];
      index /= dimSize[i];
    }
    return result;
  }

  using Iterator = tools::CountingIterator<MDSpan, decltype(&MDSpan::createIndex)>;

  __PNI_CUDA_MACRO__ Iterator begin() const { return Iterator(*this, &MDSpan::createIndex, 0); }
  __PNI_CUDA_MACRO__ Iterator end() const {
    int64_t end_fake = totalSize();
    return Iterator(*this, &MDSpan::createIndex, end_fake);
  }

  __PNI_CUDA_MACRO__ bool operator==(
      const MDSpan &other) const {
    return dimSize == other.dimSize;
  }

  template <int M>
  __PNI_CUDA_MACRO__ auto right_shrink() const {
    return MDSpan<N - M>::create(dimSize.template right_shrink<M>());
  }
  template <int M>
  __PNI_CUDA_MACRO__ auto left_shrink() const {
    return MDSpan<N - M>::create(dimSize.template left_shrink<M>());
  }
};
} // namespace openpni::experimental::core
namespace openpni::experimental::core {
template <int N>
struct MDBeginEndSpan {
  using index_type = Vector<int64_t, N>;

  Vector<int64_t, N> begins;
  Vector<int64_t, N> ends;
  __PNI_CUDA_MACRO__ static constexpr auto create(
      Vector<int64_t, N> __begin, Vector<int64_t, N> __end) {
    MDBeginEndSpan result;
    result.begins = __begin;
    result.ends = __end;
    return result;
  }
  __PNI_CUDA_MACRO__ static constexpr auto create_from_center_size(
      Vector<int64_t, N> center, Vector<int64_t, N> size) {
    MDBeginEndSpan result;
    for (int i = 0; i < N; i++) {
      result.begins[i] = center[i] - size[i] / 2;
      result.ends[i] = center[i] + (size[i] + 1) / 2;
    }
    return result;
  }
  __PNI_CUDA_MACRO__ static constexpr auto create_from_origin_size(
      Vector<int64_t, N> size) {
    return create_from_center_size(Vector<int64_t, N>::create(), size);
  }

  __PNI_CUDA_MACRO__ constexpr int64_t totalSize() const {
    int64_t result = 1;
    for (int i = 0; i < N; i++)
      result *= (ends[i] - begins[i]);
    return result;
  }

  __PNI_CUDA_MACRO__ static Vector<int64_t, N> createIndex(
      const MDBeginEndSpan &_this, int64_t index) {
    Vector<int64_t, N> result = Vector<int64_t, N>::create();
    auto dimSize = _this.ends - _this.begins;
    for (int i = 0; i < N; i++) {
      result[i] = index % dimSize[i];
      index /= dimSize[i];
    }
    for (int i = 0; i < N; i++)
      result[i] = result[i] + _this.begins[i];
    return result;
  }

  using Iterator = tools::CountingIterator<MDBeginEndSpan, decltype(&MDBeginEndSpan::createIndex)>;
  __PNI_CUDA_MACRO__ Iterator begin() const { return Iterator(*this, &MDBeginEndSpan::createIndex, 0); }
  __PNI_CUDA_MACRO__ Iterator end() const {
    int64_t end_fake = totalSize();
    return Iterator(*this, &MDBeginEndSpan::createIndex, end_fake);
  }

  __PNI_CUDA_MACRO__ bool inBounds(
      const Vector<int64_t, N> &__index) const {
    for (int i = 0; i < N; i++)
      if (__index[i] < begins[i] || __index[i] >= ends[i])
        return false;
    return true;
  }

  template <typename... Args>
  __PNI_CUDA_MACRO__ bool inBounds(
      Args... __extents) const {
    return inBounds(Vector<int64_t, N>::create(__extents...));
  }

  __PNI_CUDA_MACRO__ auto operator()(
      Vector<int64_t, N> __index) const {
    int64_t result = __index[N - 1] - begins[N - 1];
    for (int i = N - 2; i >= 0; i--)
      result = result * (ends[i] - begins[i]) + __index[i] - begins[i];
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

  __PNI_CUDA_MACRO__ Vector<int64_t, N> toIndex(
      int64_t index) const {
    Vector<int64_t, N> result = Vector<int64_t, N>::create();
    auto dimSize = ends - begins;
    for (int i = 0; i < N; i++) {
      result[i] = index % dimSize[i];
      index /= dimSize[i];
    }
    for (int i = 0; i < N; i++)
      result[i] = result[i] + begins[i];
    return result;
  }

  __PNI_CUDA_MACRO__ auto size() const { return ends - begins; }
  __PNI_CUDA_MACRO__ auto toSpan() const { return MDSpan<N>::create(ends - begins); }
};
} // namespace openpni::experimental::core
namespace openpni {
using Span3 = openpni::experimental::core::MDSpan<3>;
using Span2 = openpni::experimental::core::MDSpan<2>;
} // namespace openpni
