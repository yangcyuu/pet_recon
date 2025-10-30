#pragma once
#include <cmath>

#include "BasicMath.hpp"
namespace openpni::experimental::core {
template <typename T>
struct Vec3 {
  typedef T value_type;

  T x;
  T y;
  T z;

  template <typename TT = T>
  __PNI_CUDA_MACRO__ auto operator+(
      const Vec3<TT> &b) const -> Vec3<decltype(x + b.x)> {
    Vec3<decltype(x + b.x)> temp;
    temp.x = x + b.x;
    temp.y = y + b.y;
    temp.z = z + b.z;
    return temp;
  }
  template <typename TT = T>
  __PNI_CUDA_MACRO__ auto operator+(
      const TT &v) const -> Vec3<decltype(x + v)> {
    Vec3<decltype(x + v)> temp;
    temp.x = x + v;
    temp.y = y + v;
    temp.z = z + v;
    return temp;
  }

  template <typename TT = T>
  __PNI_CUDA_MACRO__ auto operator-(
      const Vec3<TT> &b) const -> Vec3<decltype(x - b.x)> {
    Vec3<decltype(x - b.x)> temp;
    temp.x = x - b.x;
    temp.y = y - b.y;
    temp.z = z - b.z;
    return temp;
  }

  template <typename TT = T>
  __PNI_CUDA_MACRO__ auto operator-(
      const TT &v) const -> Vec3<decltype(x - v)> {
    Vec3<decltype(x - v)> temp;
    temp.x = x - v;
    temp.y = y - v;
    temp.z = z - v;
    return temp;
  }

  template <typename TT = T>
  __PNI_CUDA_MACRO__ auto operator*(
      const TT &b) const -> Vec3<decltype(x * b)> {
    Vec3<decltype(x * b)> temp;
    temp.x = x * b;
    temp.y = y * b;
    temp.z = z * b;
    return temp;
  }

  template <typename TT = T>
  __PNI_CUDA_MACRO__ auto operator*(
      const Vec3<TT> &b) const -> Vec3<decltype(x * b.x)> {
    return pointWiseMul(b);
  }

  template <typename TT = T>
  __PNI_CUDA_MACRO__ Vec3<T> &operator+=(
      const TT &a) {
    x += a;
    y += a;
    z += a;
    return *this;
  }
  template <typename TT = T>
  __PNI_CUDA_MACRO__ Vec3<T> &operator+=(
      const Vec3<TT> &b) {
    x += b.x;
    y += b.y;
    z += b.z;
    return *this;
  }
  template <typename TT = T>
  __PNI_CUDA_MACRO__ Vec3<T> &operator-=(
      const TT &a) {
    x -= a;
    y -= a;
    z -= a;
    return *this;
  }
  template <typename TT = T>
  __PNI_CUDA_MACRO__ Vec3<T> &operator-=(
      const Vec3<TT> &b) {
    x -= b.x;
    y -= b.y;
    z -= b.z;
    return *this;
  }

  template <typename TT = T>
  __PNI_CUDA_MACRO__ Vec3<T> &operator*=(
      const TT &b) {
    x *= b;
    y *= b;
    z *= b;
    return *this;
  }

  template <typename TT = T>
  __PNI_CUDA_MACRO__ auto operator/(
      const Vec3<TT> &b) const -> Vec3<decltype(x / b.x)> {
    Vec3<decltype(x / b.x)> temp;
    temp.x = x / b.x;
    temp.y = y / b.y;
    temp.z = z / b.z;
    return temp;
  }

  template <typename TT = T>
  __PNI_CUDA_MACRO__ auto operator/(
      const TT &b) const -> Vec3<decltype(x / b)> {
    Vec3<decltype(x / b)> temp;
    temp.x = x / b;
    temp.y = y / b;
    temp.z = z / b;
    return temp;
  }

  template <typename TT = T>
  __PNI_CUDA_MACRO__ Vec3<T> &operator/=(
      const TT &b) {
    x /= b;
    y /= b;
    z /= b;
    return *this;
  }

  template <typename TT = T>
  __PNI_CUDA_MACRO__ auto pointWiseMul(
      const Vec3<TT> &b) const -> Vec3<decltype(x * b.x)> {
    Vec3<decltype(x * b.x)> temp;
    temp.x = x * b.x;
    temp.y = y * b.y;
    temp.z = z * b.z;
    return temp;
  }

  __PNI_CUDA_MACRO__ T dot(
      const Vec3<T> &b) const {
    return x * b.x + y * b.y + z * b.z;
  }

  __PNI_CUDA_MACRO__ T l22() const { return x * x + y * y + z * z; }
  __PNI_CUDA_MACRO__ auto l2() const -> std::conditional_t<std::is_same_v<T, float>, float, double> {
    if constexpr (std::is_same_v<T, float>)
      return basic::FMath<float>::fsqrt(l22());
    else
      return basic::FMath<double>::fsqrt(l22());
  }

  __PNI_CUDA_MACRO__ T l1() const {
    if constexpr (std::is_unsigned_v<T>)
      return x + y + z;
    else
      return std::abs(x) + std::abs(y) + std::abs(z);
  }
  __PNI_CUDA_MACRO__ T lmax() const {
    if constexpr (std::is_unsigned_v<T>)
      return std::max(std::max(x, y), z);
    else
      return std::max(std::max(std::abs(x), std::abs(y)), std::abs(z));
  }
  __PNI_CUDA_MACRO__ T minElement() const { return std::min(std::min(x, y), z); }
  __PNI_CUDA_MACRO__ T maxElement() const { return std::max(std::max(x, y), z); }
  template <typename TT>
  __PNI_CUDA_MACRO__ auto cross(
      const Vec3<TT> &b) const -> Vec3<decltype(y * b.z)> {
    Vec3<T> temp;
    temp.x = y * b.z - z * b.y;
    temp.y = z * b.x - x * b.z;
    temp.z = x * b.y - y * b.x;
    return temp;
  }

  __PNI_CUDA_MACRO__ auto normalized() const { return *this / l2(); }
};

template <typename T>
static __PNI_CUDA_MACRO__ Vec3<T> make_vec3(
    const T __x, const T __y, const T __z) {
  Vec3<T> temp;
  temp.x = __x;
  temp.y = __y;
  temp.z = __z;
  return temp;
}

template <typename TT, typename T>
static __PNI_CUDA_MACRO__ Vec3<TT> make_vec3(
    const Vec3<T> &__p) {
  if constexpr (std::is_same_v<T, TT>)
    return __p;

  Vec3<TT> temp;
  temp.x = __p.x;
  temp.y = __p.y;
  temp.z = __p.z;
  return temp;
}

template <typename T>
struct Vec2 {
  typedef T value_type;

  T x;
  T y;

  template <typename TT = T>
  __PNI_CUDA_MACRO__ Vec2<T> &operator=(
      const Vec2<TT> &b) {
    x = b.x;
    y = b.y;
    return *this;
  }

  template <typename TT = T>
  __PNI_CUDA_MACRO__ auto operator+(
      const Vec2<TT> &b) const -> Vec2<decltype(x + b.x)> {
    Vec2<decltype(x + b.x)> temp;
    temp.x = x + b.x;
    temp.y = y + b.y;
    return temp;
  }

  template <typename TT = T>
  __PNI_CUDA_MACRO__ auto operator+(
      const TT &b) const -> Vec2<decltype(x + b)> {
    Vec2<decltype(x + b)> temp;
    temp.x = x + b;
    temp.y = y + b;
    return temp;
  }

  template <typename TT = T>
  __PNI_CUDA_MACRO__ Vec2<T> &operator+=(
      const Vec2<TT> &b) {
    x += b.x;
    y += b.y;
    return *this;
  }

  template <typename TT = T>
  __PNI_CUDA_MACRO__ auto operator-(
      const Vec2<TT> &b) const -> Vec2<decltype(x - b.x)> {
    Vec2<decltype(x - b.x)> temp;
    temp.x = x - b.x;
    temp.y = y - b.y;
    return temp;
  }

  template <typename TT = T>
  __PNI_CUDA_MACRO__ auto operator-(
      const T &b) const -> Vec2<decltype(x - b)> {
    Vec2<decltype(x - b)> temp;
    temp.x = x - b;
    temp.y = y - b;
    return temp;
  }

  Vec2<T> operator-() const {
    Vec2<T> temp;
    temp.x = -x;
    temp.y = -y;
    return temp;
  }

  template <typename TT = T>
  __PNI_CUDA_MACRO__ auto operator*(
      const TT &b) const -> Vec2<decltype(x * b)> {
    Vec2<decltype(x * b)> temp;
    temp.x = x * b;
    temp.y = y * b;
    return temp;
  }

  template <typename TT = T>
  __PNI_CUDA_MACRO__ auto operator*(
      const Vec2<TT> &b) const -> Vec2<decltype(x * b.x)> {
    Vec2<decltype(x * b.x)> temp;
    temp.x = x * b.x;
    temp.y = y * b.y;
    return temp;
  }

  template <typename TT = T>
  __PNI_CUDA_MACRO__ auto operator/(
      const Vec2<TT> &b) const -> Vec2<decltype(x / b.x)> {
    Vec2<decltype(x / b.x)> temp;
    temp.x = x / b.x;
    temp.y = y / b.y;
    return temp;
  }

  template <typename TT = T>
  __PNI_CUDA_MACRO__ auto operator/(
      const TT &b) const -> Vec2<decltype(x / b)> {
    Vec2<decltype(x / b)> temp;
    temp.x = x / b;
    temp.y = y / b;
    return temp;
  }

  template <typename TT = T>
  __PNI_CUDA_MACRO__ auto pointWiseMul(
      const Vec2<TT> &b) const -> Vec2<decltype(x * b.x)> {
    Vec2<decltype(x * b.x)> temp;
    temp.x = x * b.x;
    temp.y = y * b.y;
    return temp;
  }

  __PNI_CUDA_MACRO__ Vec2<T> dot(
      const Vec2<T> &b) const {
    return x * b.x + y * b.y;
  }

  __PNI_CUDA_MACRO__ double l22() const { return x * x + y * y; }

  __PNI_CUDA_MACRO__ double l2() const { return sqrt(l22()); }
};

template <typename T, typename TT = T>
static __PNI_CUDA_MACRO__ Vec2<T> make_vec2(
    T __x, TT __y) {
  Vec2<T> temp;
  temp.x = __x;
  temp.y = __y;
  return temp;
}
template <typename T, typename TT = T>
static __PNI_CUDA_MACRO__ Vec2<T> make_vec2(
    const Vec2<TT> &__p) {
  Vec2<T> temp;
  temp.x = __p.x;
  temp.y = __p.y;
  return temp;
}

template <typename T>
inline __PNI_CUDA_MACRO__ T det3(
    const Vec3<T> U, const Vec3<T> V, const Vec3<T> W) {
  return U.x * (V.y * W.z - V.z * W.y) - U.y * (V.x * W.z - V.z * W.x) + U.z * (V.x * W.y - V.y * W.x);
}

template <FloatingPoint_c T>
inline __PNI_CUDA_MACRO__ Vec2<T> point_ceil(
    const Vec2<T> &__p) {
  Vec2<T> temp;
  temp.x = ceil(__p.x);
  temp.y = ceil(__p.y);
  return temp;
}
template <FloatingPoint_c T>
inline __PNI_CUDA_MACRO__ Vec3<T> point_ceil(
    const Vec3<T> &__p) {
  Vec3<T> temp;
  temp.x = ceil(__p.x);
  temp.y = ceil(__p.y);
  temp.z = ceil(__p.z);
  return temp;
}

template <FloatingPoint_c T>
inline __PNI_CUDA_MACRO__ Vec2<T> point_floor(
    const Vec2<T> &__p) {
  Vec2<T> temp;
  temp.x = floor(__p.x);
  temp.y = floor(__p.y);
  return temp;
}
template <FloatingPoint_c T>
inline __PNI_CUDA_MACRO__ Vec3<T> point_floor(
    const Vec3<T> &__p) {
  Vec3<T> temp;
  temp.x = floor(__p.x);
  temp.y = floor(__p.y);
  temp.z = floor(__p.z);
  return temp;
}

}; // namespace openpni::experimental::core
namespace openpni::experimental {
// template <typename T>
// concept FloatingPoint2D_c = std::same_as<experimental::core::Vec2<typename T::value_type>, T> &&
//                             experimental::FloatingPoint_c<typename T::value_type>;
// template <typename T>
// concept FloatingPoint3D_c = std::same_as<experimental::core::Vec3<typename T::value_type>, T> &&
//                             experimental::FloatingPoint_c<typename T::value_type>;

using p3di = core::Vec3<int>;
using p3df = core::Vec3<float>;
using p3dd = core::Vec3<double>;
using p2df = core::Vec2<float>;
using p2dd = core::Vec2<double>;
using cubei = core::Vec2<p3di>;
using cubef = core::Vec2<p3df>;
using cubed = core::Vec2<p3dd>;
template <FloatingPoint_c T>
using cube = core::Vec2<core::Vec3<T>>;
} // namespace openpni::experimental
