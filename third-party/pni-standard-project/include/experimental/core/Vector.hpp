#pragma once
#include <cmath>

#include "BasicMath.hpp"
namespace openpni::experimental::core {
template <typename T, int N>
  requires(N >= 1)
struct Vector {
  T data[N];
  __PNI_CUDA_MACRO__ static constexpr Vector create() { return Vector(); }
  __PNI_CUDA_MACRO__ static constexpr Vector create(
      T v) {
    Vector result;
    for (int i = 0; i < N; i++)
      result.data[i] = v;
    return result;
  }

  template <typename... Args>
    requires(sizeof...(Args) >= 2)
  __PNI_CUDA_MACRO__ static constexpr Vector create(
      Args... args) {
    static_assert(sizeof...(Args) == N, "Number of arguments must match vector dimension");
    return Vector{static_cast<T>(args)...};
  }
  template <typename... Args>
  __PNI_CUDA_MACRO__ constexpr Vector<T, N + sizeof...(Args)> right_expand(
      Args... args) const {
    Vector<T, N + sizeof...(Args)> result;
    for (int i = 0; i < N; i++)
      result.data[i] = data[i];
    int idx = N;
    ((result.data[idx++] = static_cast<T>(args)), ...);
    return result;
  }
  template <typename... Args>
  __PNI_CUDA_MACRO__ constexpr Vector<T, N + sizeof...(Args)> left_expand(
      Args... args) const {
    Vector<T, N + sizeof...(Args)> result;
    int idx = 0;
    ((result.data[idx++] = static_cast<T>(args)), ...);
    for (int i = 0; i < N; i++)
      result.data[idx++] = data[i];
    return result;
  }

  template <int M>
    requires(M > 0 && M < N)
  __PNI_CUDA_MACRO__ constexpr Vector<T, N - M> right_shrink() const {
    Vector<T, N - M> result;
    for (int i = 0; i < N - M; i++)
      result.data[i] = data[i];
    return result;
  }
  template <int M>
    requires(M > 0 && M < N)
  __PNI_CUDA_MACRO__ constexpr Vector<T, N - M> left_shrink() const {
    Vector<T, N - M> result;
    for (int i = 0; i < N - M; i++)
      result.data[i] = data[i + M];
    return result;
  }
  template <int M>
  __PNI_CUDA_MACRO__ constexpr Vector<T, N + M> right_expand(
      const Vector<T, M> &other) const {
    Vector<T, N + M> result;
    for (int i = 0; i < N; i++)
      result.data[i] = data[i];
    for (int i = 0; i < M; i++)
      result.data[i + N] = other.data[i];
    return result;
  }
  template <int M>
  __PNI_CUDA_MACRO__ constexpr Vector<T, N + M> left_expand(
      const Vector<T, M> &other) const {
    Vector<T, N + M> result;
    for (int i = 0; i < M; i++)
      result.data[i] = other.data[i];
    for (int i = 0; i < N; i++)
      result.data[i + M] = data[i];
    return result;
  }

  template <typename TT>
  __PNI_CUDA_MACRO__ constexpr Vector<TT, N> to() const {
    if constexpr (std::is_same_v<T, TT>)
      return *this;
    else {
      Vector<TT, N> result;
      for (int i = 0; i < N; i++)
        result.data[i] = static_cast<TT>(data[i]);
      return result;
    }
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

public: // Addition operators
  template <typename TT = T>
  __PNI_CUDA_MACRO__ auto operator+(
      const Vector<TT, N> &other) const -> Vector<decltype(TT() + T()), N> {
    Vector<decltype(TT() + T()), N> result;
    for (int i = 0; i < N; i++)
      result[i] = data[i] + other[i];
    return result;
  }
  template <typename TT = T>
  __PNI_CUDA_MACRO__ auto operator+(
      TT v) const -> Vector<decltype(TT() + T()), N> {
    Vector<decltype(TT() + T()), N> result;
    for (int i = 0; i < N; i++)
      result[i] = data[i] + v;
    return result;
  }
  template <typename TT>
    requires(std::is_arithmetic_v<TT>)
  __PNI_CUDA_MACRO__ friend auto operator+(
      TT v, const Vector<T, N> &vec) -> Vector<decltype(TT() + T()), N> {
    Vector<decltype(TT() + T()), N> result;
    for (int i = 0; i < N; i++)
      result[i] = vec[i] + v;
    return result;
  }
  __PNI_CUDA_MACRO__ Vector operator+() const { return *this; }

public: // Subtraction operators
  template <typename TT = T>
  __PNI_CUDA_MACRO__ auto operator-(
      const Vector<TT, N> &other) const -> Vector<decltype(TT() - T()), N> {
    Vector<decltype(TT() - T()), N> result;
    for (int i = 0; i < N; i++)
      result[i] = data[i] - other[i];
    return result;
  }
  template <typename TT = T>
  __PNI_CUDA_MACRO__ auto operator-(
      TT v) const -> Vector<decltype(TT() - T()), N> {
    Vector<decltype(TT() - T()), N> result;
    for (int i = 0; i < N; i++)
      result[i] = data[i] - v;
    return result;
  }
  template <typename TT>
    requires(std::is_arithmetic_v<TT>)
  __PNI_CUDA_MACRO__ friend auto operator-(
      TT v, const Vector<T, N> &vec) -> Vector<decltype(TT() - T()), N> {
    Vector<decltype(TT() - T()), N> result;
    for (int i = 0; i < N; i++)
      result[i] = v - vec[i];
    return result;
  }
  __PNI_CUDA_MACRO__ Vector operator-() const {
    Vector result;
    for (int i = 0; i < N; i++)
      result[i] = -data[i];
    return result;
  }

public: // Multiplication operators
  template <typename TT = T>
  __PNI_CUDA_MACRO__ auto operator*(
      const Vector<TT, N> &other) const -> Vector<decltype(TT() * T()), N> {
    Vector<decltype(TT() * T()), N> result;
    for (int i = 0; i < N; i++)
      result[i] = data[i] * other[i];
    return result;
  }
  template <typename TT = T>
  __PNI_CUDA_MACRO__ auto operator*(
      TT v) const -> Vector<decltype(TT() * T()), N> {
    Vector<decltype(TT() * T()), N> result;
    for (int i = 0; i < N; i++)
      result[i] = data[i] * v;
    return result;
  }
  template <typename TT>
    requires(std::is_arithmetic_v<TT>)
  __PNI_CUDA_MACRO__ friend auto operator*(
      TT v, const Vector<T, N> &vec) -> Vector<decltype(TT() * T()), N> {
    Vector<decltype(TT() * T()), N> result;
    for (int i = 0; i < N; i++)
      result[i] = vec[i] * v;
    return result;
  }

public: // Division operators
  template <typename TT = T>
  __PNI_CUDA_MACRO__ auto operator/(
      const Vector<TT, N> &other) const -> Vector<decltype(TT() / T()), N> {
    Vector<decltype(TT() / T()), N> result;
    for (int i = 0; i < N; i++)
      result[i] = data[i] / other[i];
    return result;
  }
  template <typename TT = T>
  __PNI_CUDA_MACRO__ auto operator/(
      TT v) const -> Vector<decltype(TT() / T()), N> {
    Vector<decltype(TT() / T()), N> result;
    for (int i = 0; i < N; i++)
      result[i] = data[i] / v;
    return result;
  }
  template <typename TT>
    requires(std::is_arithmetic_v<TT>)
  __PNI_CUDA_MACRO__ friend auto operator/(
      TT v, const Vector<T, N> &vec) -> Vector<decltype(TT() / T()), N> {
    Vector<decltype(TT() / T()), N> result;
    for (int i = 0; i < N; i++)
      result[i] = v / vec[i];
    return result;
  }

public: // Other operators
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
  __PNI_CUDA_MACRO__ T lmul() const {
    T result = 1;
    for (int i = 0; i < N; i++)
      result *= data[i];
    return result;
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

  auto reverse() const {
    Vector<T, N> result;
    for (int i = 0; i < N; i++)
      result[i] = data[N - 1 - i];
    return result;
  }

  bool operator==(
      const Vector &other) const {
    for (int i = 0; i < N; i++)
      if (data[i] != other.data[i])
        return false;
    return true;
  }
  bool operator!=(
      const Vector &other) const {
    return !(*this == other);
  }
};

template <std::size_t Index, typename T, int N>
__PNI_CUDA_MACRO__ constexpr T &get(
    Vector<T, N> &vec) {
  static_assert(Index < static_cast<std::size_t>(N), "Index out of range for Vector<T, N>");
  return vec[Index];
}

template <std::size_t Index, typename T, int N>
__PNI_CUDA_MACRO__ constexpr const T &get(
    const Vector<T, N> &vec) {
  static_assert(Index < static_cast<std::size_t>(N), "Index out of range for Vector<T, N>");
  return vec[Index];
}

template <std::size_t Index, typename T, int N>
__PNI_CUDA_MACRO__ constexpr T &&get(
    Vector<T, N> &&vec) {
  static_assert(Index < static_cast<std::size_t>(N), "Index out of range for Vector<T, N>");
  return std::move(vec[Index]);
}

template <std::size_t Index, typename T, int N>
__PNI_CUDA_MACRO__ constexpr const T &&get(
    const Vector<T, N> &&vec) {
  static_assert(Index < static_cast<std::size_t>(N), "Index out of range for Vector<T, N>");
  return std::move(vec[Index]);
}

} // namespace openpni::experimental::core
namespace openpni::experimental::algorithms {
template <typename T>
__PNI_CUDA_MACRO__ core::Vector<T, 3> cross(
    const core::Vector<T, 3> &a, const core::Vector<T, 3> &b) {
  core::Vector<T, 3> result;
  result[0] = a[1] * b[2] - a[2] * b[1];
  result[1] = a[2] * b[0] - a[0] * b[2];
  result[2] = a[0] * b[1] - a[1] * b[0];
  return result;
}
template <Arithmetic_c T, int N>
__PNI_CUDA_MACRO__ auto l2(
    core::Vector<T, N> const &v) {
  if constexpr (std::is_same_v<T, float>)
    return core::FMath<float>::fsqrt(v.l22());
  else
    return core::FMath<double>::fsqrt(v.l22());
}
template <Arithmetic_c T, int N>
__PNI_CUDA_MACRO__ auto normalized(
    core::Vector<T, N> const &v) {
  return v / l2(v);
}
template <typename T, int N, typename Func>
__PNI_CUDA_HOST_ONLY__ constexpr auto apply(
    core::Vector<T, N> const &v, Func &&op) {
  core::Vector<decltype(op(T())), N> result;
  for (int i = 0; i < N; i++)
    result.data[i] = op(v[i]);
  return result;
}
template <typename T, int N, typename Func>
__PNI_CUDA_HOST_ONLY__ constexpr auto reduce(
    core::Vector<T, N> const &v, Func &&op, auto init) {
  using RT = decltype(op(T(), T()));
  RT result = static_cast<RT>(init);
  for (int i = 0; i < N; i++)
    result = op(result, v[i]);
  return result;
}

} // namespace openpni::experimental::algorithms

namespace openpni {} // namespace openpni
namespace openpni::experimental {
using p3df = openpni::experimental::core::Vector<float, 3>;
using p2df = openpni::experimental::core::Vector<float, 2>;
using p3di = openpni::experimental::core::Vector<int, 3>;
using p3dli = openpni::experimental::core::Vector<int64_t, 3>;
using p2di = openpni::experimental::core::Vector<int, 2>;
using p2dli = openpni::experimental::core::Vector<int64_t, 2>;
using p3dd = openpni::experimental::core::Vector<double, 3>;
using p2dd = openpni::experimental::core::Vector<double, 2>;
using cubef = openpni::experimental::core::Vector<p3df, 2>;
using cubed = openpni::experimental::core::Vector<p3dd, 2>;
template <typename T>
using Vec3 = openpni::experimental::core::Vector<T, 3>;
template <typename T>
using Vec2 = openpni::experimental::core::Vector<T, 2>;
template <FloatingPoint_c T>
using cube = openpni::experimental::core::Vector<openpni::experimental::core::Vector<T, 3>, 2>;
} // namespace openpni::experimental

namespace std {
template <typename T, int N>
struct tuple_size<openpni::experimental::core::Vector<T, N>>
    : std::integral_constant<std::size_t, static_cast<std::size_t>(N)> {};

template <std::size_t Index, typename T, int N>
struct tuple_element<Index, openpni::experimental::core::Vector<T, N>> {
  static_assert(Index < static_cast<std::size_t>(N), "Index out of range for Vector<T, N>");
  using type = T;
};
} // namespace std