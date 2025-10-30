#pragma once
#include <atomic>

#include "../basic/Image.hpp"
#include "../basic/Math.hpp"
#include "../basic/Point.hpp"
// 3D Interpolation Methods
namespace openpni::math {
struct InterpolationNearest3D {
  template <typename ImageType, FloatingPoint_c T>
  __PNI_CUDA_MACRO__ ImageType operator()(
      const basic::Vec3<T> &point, const ImageType *img, const basic::Image3DGeometry &geometry) const {
    const auto integerIndex = basic::make_vec3<unsigned>((point - geometry.imgBegin) / geometry.voxelSize);
    if (geometry.in(integerIndex))
      return img[geometry.at(integerIndex)];
    return 0;
  }

  template <typename ImageType, FloatingPoint_c T>
  __PNI_CUDA_MACRO__ ImageType get_value(
      const basic::Vec3<T> &point, const ImageType *img, const basic::Image3DGeometry &geometry) const {
    return operator()(point, img, geometry);
  }

  template <typename ImageType, FloatingPoint_c T>
  __PNI_CUDA_MACRO__ void add_value(
      const basic::Vec3<T> &point, ImageType *img, const ImageType &fwd, const basic::Image3DGeometry &geometry) const {
    const auto integerIndex = basic::make_vec3<unsigned>((point - geometry.imgBegin) / geometry.voxelSize);
#ifdef __CUDA_ARCH__
    if (geometry.in(integerIndex))
      atomicAdd(&img[geometry.at(integerIndex)], fwd);
#else
    if (geometry.in(integerIndex))
      std::atomic_ref(img[geometry.at(integerIndex)]) += fwd;
#endif
  }
};

struct InterpolationTrilinear3D {
  template <typename ImageType, FloatingPoint_c T>
  __PNI_CUDA_MACRO__ ImageType operator()(
      const basic::Vec3<T> &point, const ImageType *img, const basic::Image3DGeometry &geometry) const {
    const auto index = (point - geometry.imgBegin) / geometry.voxelSize - T(0.5);
    const auto i000 = basic::make_vec3<int>(basic::point_floor(index));

    const basic::Vec3<T> cubeBegin = geometry.imgBegin + geometry.voxelSize.pointWiseMul(i000 + T(0.5));
    const basic::Vec3<T> cubeEnd =
        geometry.imgBegin + geometry.voxelSize.pointWiseMul((i000 + basic::make_vec3<int>(1, 1, 1) + T(0.5)));
    const basic::Vec3<T> uniformedPoint = (point - cubeBegin) / (cubeEnd - cubeBegin);

    const auto i100 = i000 + basic::make_vec3<int>(1, 0, 0);
    const auto i010 = i000 + basic::make_vec3<int>(0, 1, 0);
    const auto i110 = i000 + basic::make_vec3<int>(1, 1, 0);
    const auto i001 = i000 + basic::make_vec3<int>(0, 0, 1);
    const auto i101 = i000 + basic::make_vec3<int>(1, 0, 1);
    const auto i011 = i000 + basic::make_vec3<int>(0, 1, 1);
    const auto i111 = i000 + basic::make_vec3<int>(1, 1, 1);

    T _000 = geometry.in(i000) ? img[geometry.at(i000)] : 0;
    T _100 = geometry.in(i100) ? img[geometry.at(i100)] : 0;
    T _010 = geometry.in(i010) ? img[geometry.at(i010)] : 0;
    T _110 = geometry.in(i110) ? img[geometry.at(i110)] : 0;
    T _001 = geometry.in(i001) ? img[geometry.at(i001)] : 0;
    T _101 = geometry.in(i101) ? img[geometry.at(i101)] : 0;
    T _011 = geometry.in(i011) ? img[geometry.at(i011)] : 0;
    T _111 = geometry.in(i111) ? img[geometry.at(i111)] : 0;

    _000 *= (1 - uniformedPoint.x) * (1 - uniformedPoint.y) * (1 - uniformedPoint.z);
    _100 *= uniformedPoint.x * (1 - uniformedPoint.y) * (1 - uniformedPoint.z);
    _010 *= (1 - uniformedPoint.x) * uniformedPoint.y * (1 - uniformedPoint.z);
    _110 *= uniformedPoint.x * uniformedPoint.y * (1 - uniformedPoint.z);
    _001 *= (1 - uniformedPoint.x) * (1 - uniformedPoint.y) * uniformedPoint.z;
    _101 *= uniformedPoint.x * (1 - uniformedPoint.y) * uniformedPoint.z;
    _011 *= (1 - uniformedPoint.x) * uniformedPoint.y * uniformedPoint.z;
    _111 *= uniformedPoint.x * uniformedPoint.y * uniformedPoint.z;

    return (_000 + _100 + _010 + _110 + _001 + _101 + _011 + _111);
  }

  template <typename ImageType, FloatingPoint_c T>
  __PNI_CUDA_MACRO__ ImageType get_value(
      const basic::Vec3<T> &point, const ImageType *img, const basic::Image3DGeometry &geometry) const {
    return operator()(point, img, geometry);
  }

  template <typename ImageType, FloatingPoint_c T>
  __PNI_CUDA_MACRO__ void add_value(
      const basic::Vec3<T> &point, ImageType *img, const ImageType &fwd, const basic::Image3DGeometry &geometry) const {
    const auto index = (point - geometry.imgBegin) / geometry.voxelSize - T(0.5);
    const auto i000 = basic::make_vec3<int>(basic::point_floor(index));

    const basic::Vec3<T> cubeBegin = geometry.imgBegin + geometry.voxelSize.pointWiseMul(i000 + T(0.5));
    const basic::Vec3<T> cubeEnd =
        geometry.imgBegin + geometry.voxelSize.pointWiseMul((i000 + basic::make_vec3<int>(1, 1, 1) + T(0.5)));
    const basic::Vec3<T> uniformedPoint = (point - cubeBegin) / (cubeEnd - cubeBegin);

    const auto i100 = i000 + basic::make_vec3<int>(1, 0, 0);
    const auto i010 = i000 + basic::make_vec3<int>(0, 1, 0);
    const auto i110 = i000 + basic::make_vec3<int>(1, 1, 0);
    const auto i001 = i000 + basic::make_vec3<int>(0, 0, 1);
    const auto i101 = i000 + basic::make_vec3<int>(1, 0, 1);
    const auto i011 = i000 + basic::make_vec3<int>(0, 1, 1);
    const auto i111 = i000 + basic::make_vec3<int>(1, 1, 1);

#ifdef __CUDA_ARCH__
#define atomic_add(A, B) atomicAdd(&A, B)
#else
#define atomic_add(A, B) std::atomic_ref(A) += B
#endif
    if (geometry.in(i000)) // 000
      atomic_add(img[geometry.at(i000)],
                 fwd * (1 - uniformedPoint.x) * (1 - uniformedPoint.y) * (1 - uniformedPoint.z));
    if (geometry.in(i100)) // 100
      atomic_add(img[geometry.at(i100)], fwd * uniformedPoint.x * (1 - uniformedPoint.y) * (1 - uniformedPoint.z));
    if (geometry.in(i010)) // 010
      atomic_add(img[geometry.at(i010)], fwd * (1 - uniformedPoint.x) * uniformedPoint.y * (1 - uniformedPoint.z));
    if (geometry.in(i110)) // 110
      atomic_add(img[geometry.at(i110)], fwd * uniformedPoint.x * uniformedPoint.y * (1 - uniformedPoint.z));
    if (geometry.in(i001)) // 001
      atomic_add(img[geometry.at(i001)], fwd * (1 - uniformedPoint.x) * (1 - uniformedPoint.y) * uniformedPoint.z);
    if (geometry.in(i101)) // 101
      atomic_add(img[geometry.at(i101)], fwd * uniformedPoint.x * (1 - uniformedPoint.y) * uniformedPoint.z);
    if (geometry.in(i011)) // 011
      atomic_add(img[geometry.at(i011)], fwd * (1 - uniformedPoint.x) * uniformedPoint.y * uniformedPoint.z);
    if (geometry.in(i111)) // 111
      atomic_add(img[geometry.at(i111)], fwd * uniformedPoint.x * uniformedPoint.y * uniformedPoint.z);
#undef atomic_add
  }
};
} // namespace openpni::math

// 2D Interpolation Methods
namespace openpni::math {
template <FloatingPoint_c T>
struct InterpolationNearest2D {
  typedef T value_type;
  template <FloatingPoint_c TT = T>
  __PNI_CUDA_MACRO__ T operator()(
      const basic::Vec2<TT> &point, const T *img, const basic::Image2DGeometry &geometry) const {
    using fmath = basic::FMath<TT>;
    if (fmath::isNaN(point.x) || fmath::isNaN(point.y) || fmath::isInf(point.x) || fmath::isInf(point.y))
      return 0;

    const auto integerIndex = basic::make_vec2<unsigned>((point - geometry.imgBegin) / geometry.voxelSize);
    if (geometry.in(integerIndex))
      return img[geometry.at(integerIndex)];
    return 0;
  }
};

template <FloatingPoint_c T>
struct InterpolationBilinear2D {
  typedef T value_type;
  template <FloatingPoint_c TT = T>
  __PNI_CUDA_MACRO__ T operator()(
      const basic::Vec2<TT> &point, const T *img, const basic::Image2DGeometry &geometry) const {
    using fmath = basic::FMath<TT>;
    if (fmath::isNaN(point.x) || fmath::isNaN(point.y) || fmath::isInf(point.x) || fmath::isInf(point.y))
      return 0;

    const auto index = (point - geometry.imgBegin) / geometry.voxelSize - TT(0.5);
    const auto i00 = basic::make_vec2<int>(basic::point_floor(index));

    const basic::Vec2<TT> squareBegin =
        basic::make_vec2<TT>(geometry.imgBegin + geometry.voxelSize.pointWiseMul(i00 + TT(0.5)));
    const basic::Vec2<TT> squareEnd = basic::make_vec2<TT>(
        geometry.imgBegin + geometry.voxelSize.pointWiseMul((i00 + basic::make_vec2<int>(1, 1) + TT(0.5))));
    const basic::Vec2<TT> uniformedPoint = (point - squareBegin) / (squareEnd - squareBegin);

    const T v00 = geometry.in(i00) ? img[geometry.at(i00)] : 0;
    const T v10 =
        geometry.in(i00 + basic::make_vec2<int>(1, 0)) ? img[geometry.at(i00 + basic::make_vec2<int>(1, 0))] : 0;
    const T v01 =
        geometry.in(i00 + basic::make_vec2<int>(0, 1)) ? img[geometry.at(i00 + basic::make_vec2<int>(0, 1))] : 0;
    const T v11 =
        geometry.in(i00 + basic::make_vec2<int>(1, 1)) ? img[geometry.at(i00 + basic::make_vec2<int>(1, 1))] : 0;

    const T _00 = v00 * (1 - uniformedPoint.x) * (1 - uniformedPoint.y);
    const T _10 = v10 * uniformedPoint.x * (1 - uniformedPoint.y);
    const T _01 = v01 * (1 - uniformedPoint.x) * uniformedPoint.y;
    const T _11 = v11 * uniformedPoint.x * uniformedPoint.y;

    return (_00 + _10 + _01 + _11);
  }
};

} // namespace openpni::math