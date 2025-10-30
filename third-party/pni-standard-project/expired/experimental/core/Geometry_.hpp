#pragma once
#include <cmath>

#include "BasicMath.hpp"
#include "Image.hpp"
#include "Point.hpp"
namespace openpni::experimental::core {
template <typename T>
concept FloatingPoint2D_c = std::same_as<experimental::core::Vec2<typename T::value_type>, T> &&
                            experimental::FloatingPoint_c<typename T::value_type>;
template <typename T>
concept FloatingPoint3D_c = std::same_as<experimental::core::Vec3<typename T::value_type>, T> &&
                            experimental::FloatingPoint_c<typename T::value_type>;

using p3di = openpni::experimental::core::Vec3<int>;
using p3df = experimental::core::Vec3<float>;
using p3dd = experimental::core::Vec3<double>;
using p2df = experimental::core::Vec2<float>;
using p2dd = experimental::core::Vec2<double>;
using cubei = experimental::core::Vec2<p3di>;
using cubef = experimental::core::Vec2<p3df>;
using cubed = experimental::core::Vec2<p3dd>;
// template <experimental::FloatingPoint_c T>
// using cube = experimental::core::Vec2<experimental::core::Vec3<T>>;

// template <FloatingPoint_c T>
// __PNI_CUDA_MACRO__ core::Vec2<T> liang_barskey(
//     const cube<T> *roi, const Vec3<T> *p1, const Vec3<T> *p2) {
//   using fmath = FMath<T>;
//   const auto &imageBegin = roi->x;
//   const auto &imageEnd = roi->y;

//   const auto p2_p1 = *p2 - *p1;

//   const T ax0 = (imageBegin.x - p1->x) / p2_p1.x;
//   const T axn = (imageEnd.x - p1->x) / p2_p1.x;
//   const T axmin = fmath::min(ax0, axn);
//   const T axmax = fmath::max(ax0, axn);

//   const T ay0 = (imageBegin.y - p1->y) / p2_p1.y;
//   const T ayn = (imageEnd.y - p1->y) / p2_p1.y;
//   const T aymin = fmath::min(ay0, ayn);
//   const T aymax = fmath::max(ay0, ayn);

//   const T az0 = (imageBegin.z - p1->z) / p2_p1.z;
//   const T azn = (imageEnd.z - p1->z) / p2_p1.z;
//   const T azmin = fmath::min(az0, azn);
//   const T azmax = fmath::max(az0, azn);

//   const T amax = fmath::min(fmath::min(T(1), axmax), fmath::min(aymax, azmax));
//   const T amin = fmath::max(fmath::max(T(0), axmin), fmath::max(aymin, azmin));

//   return {amin, amax};
// }

// template <FloatingPoint_c T>
// __PNI_CUDA_MACRO__ cube<T> cube_absolute(
//     const cube<T> &__roi) // 将立方体范围转换成相同但正规的表达
// {
//   using fmath = FMath<T>;
//   return make_vec2<Vec3<T>>(make_vec3<T>(fmath::min(__roi.x.x, __roi.y.x), fmath::min(__roi.x.y, __roi.y.y),
//                                          fmath::min(__roi.x.z, __roi.y.z)),
//                             make_vec3<T>(fmath::max(__roi.x.x, __roi.y.x), fmath::max(__roi.x.y, __roi.y.y),
//                                          fmath::max(__roi.x.z, __roi.y.z)));
// }
// template <FloatingPoint_c T>
// __PNI_CUDA_MACRO__ cube<T> cube_and(
//     // 计算两个立方体的交集
//     const cube<T> &__roi1, // => roi_min
//     const cube<T> &__roi2  // => roi_max
// ) {
//   using fmath = FMath<T>;
//   return make_vec2<Vec3<T>>(make_vec3<T>(fmath::max(__roi1.x.x, __roi2.x.x), fmath::max(__roi1.x.y, __roi2.x.y),
//                                          fmath::max(__roi1.x.z, __roi2.x.z)),
//                             make_vec3<T>(fmath::min(__roi1.y.x, __roi2.y.x), fmath::min(__roi1.y.y, __roi2.y.y),
//                                          fmath::min(__roi1.y.z, __roi2.y.z)));
// }

template <FloatingPoint_c T>
struct Line {
  Vec3<T> start;
  Vec3<T> direction; // Normalization is not assumed
  __PNI_CUDA_MACRO__ static auto create_from_start_direction(
      const Vec3<T> &__start, const Vec3<T> &__direction) {
    Line<T> result;
    result.start = __start;
    result.direction = __direction;
    return result;
  }
  __PNI_CUDA_MACRO__ static auto create_from_ends(
      const Vec3<T> &__start, const Vec3<T> &__end) {
    return create_from_start_direction(__start, __end - __start);
  }
};
template <FloatingPoint_c T>
struct Plane {
  Vec3<T> start;
  Vec3<T> directionU; // Normalization is not assumed
  Vec3<T> directionV; // Normalization is not assumed
  __PNI_CUDA_MACRO__ static auto create(
      const Vec3<T> &__start, const Vec3<T> &__directionU, const Vec3<T> &__directionV) {
    Plane<T> result;
    result.start = __start;
    result.directionU = __directionU;
    result.directionV = __directionV;
    return result;
  }
};
template <FloatingPoint_c T>
struct IntersectionLinePlane {
  T t_line;
  T u_plane;
  T v_plane;
  __PNI_CUDA_MACRO__ static auto create(
      T t_line, T u_plane, T v_plane) {
    IntersectionLinePlane<T> result;
    result.t_line = t_line;
    result.u_plane = u_plane;
    result.v_plane = v_plane;
    return result;
  }
};
template <FloatingPoint_c T>
__PNI_CUDA_MACRO__ inline IntersectionLinePlane<T> intersection(
    const Line<T> &__line, const Plane<T> &__plane) {
  // 计算平面法向量
  const auto planeNormal = __plane.directionU.cross(__plane.directionV);

  // 计算直线方向向量与平面法向量的点积
  const T denom = planeNormal.dot(__line.direction);

  // 计算直线参考点到平面参考点的向量
  const auto diff = __plane.start - __line.start;

  // 计算直线参数 t_line
  const T t_line = diff.dot(planeNormal) / denom;

  // 计算交点
  const auto intersection = __line.start + __line.direction * t_line;

  // 计算平面参数 (u_plane, v_plane)
  const auto diffIntersection = intersection - __plane.start;

  // 为了减少向量长度的计算次数，直接使用点积
  const T u_plane = diffIntersection.dot(__plane.directionU) / __plane.directionU.dot(__plane.directionU);
  const T v_plane = diffIntersection.dot(__plane.directionV) / __plane.directionV.dot(__plane.directionV);

  return IntersectionLinePlane<T>::create(t_line, u_plane, v_plane);
}

template <FloatingPoint_c T>
__PNI_CUDA_MACRO__ inline T cosine(
    const Vec3<T> &__a, const Vec3<T> &__b) {
  return __a.dot(__b) / (__a.l2() * __b.l2());
}

template <FloatingPoint_c T>
__PNI_CUDA_MACRO__ inline T calculateProjectionArea(
    T planeArea, const Vec3<T> &planeNormal,
    const Vec3<T> &projectionDirection) { // 给定一个平面面积 以及平面法向量 求其在任意向量方向的投影面积

  // 计算两个向量的正弦值
  T abs_sinTheta =
      openpni::basic::abs<T>((planeNormal.dot(projectionDirection)) / (planeNormal.l2() * projectionDirection.l2()));

  // 投影面积 = 原面积 * |sin(θ)|
  return planeArea * abs_sinTheta;
}

/**
 * @brief 探测器全局坐标定义如下：
 * @note 坐标系方向定义：
 *     |法线（Z）
 *     |
 *     |
 *     ⊙----------V方向（Y）
 *    /
 *   /
 *  /
 * /U方向（X）
 * @details position: 探测器晶体外表面所在平面中心在世界坐标系中的位置
 * @details direction: 探测器晶体表面法向量在世界坐标系中的方向，法向量朝远离探测器的方向
 * @details rotation: 探测器晶体平面U方向在世界坐标系中的方向
 */
template <FloatingPoint_c T>
struct Coordinate3D {
  Vec3<T> position;
  Vec3<T> direction;
  Vec3<T> rotation;
};

/**
 * @brief 探测器局部几何结构定义
 * @note 局部坐标系定义：
 *
 *     ^ V方向（Y）
 *     |
 *     |
 *     +-------------------+
 *     |                   |
 *     |                   |
 *     |         ⊙         |  <-- 矩形中心为平面直角坐标系原点
 *     |                   |
 *     |                   |
 *     +-------------------+  --> U方向（X）
 *
 * @details UV方向是相对的，每个探测器都需要固定一套定义
 * @details 局部晶体坐标范围是[0, blockNumU * blockNumV * crystalNumU * crystalNumV)
 * @details 块内局部晶体坐标和块坐标类似，排列方式是：(u, v) =>
 * *     (0, 0), (1, 0), ..., (blockNumU - 1, 0),
 * *     (0, 1), (1, 1), ..., (blockNumU - 1, 1),
 * *     ...,
 * *     (0, blockNumV - 1), (1, blockNumV - 1), ..., (blockNumU - 1, blockNumV - 1)
 */

struct DetectorGeometry {
  uint8_t blockNumU;    // U方向的Block数量
  uint8_t blockNumV;    // V方向的Block数量
  float blockSizeU;     // U方向相邻Block中心的距离
  float blockSizeV;     // V方向相邻Block中心的距离
  uint16_t crystalNumU; // U方向Block内的晶体数量
  uint16_t crystalNumV; // V方向Block内的晶体数量
  float crystalSizeU;   // U方向相邻晶体中心的距离
  float crystalSizeV;   // V方向相邻晶体中心的距离

  uint16_t getCrystalNumInBlock() const { return crystalNumU * crystalNumV; }
  uint16_t getTotalBlockNum() const { return blockNumU * blockNumV; }
  uint16_t getTotalCrystalNum() const { return getTotalBlockNum() * getCrystalNumInBlock(); }
};

struct CrystalGeometry {
  Vec3<float> position;   // 探测器晶体外表面中心在世界坐标系中的位置
  Vec3<float> directionU; // 从探测器晶体外表面中心到U方向晶体边缘的向量
  Vec3<float> directionV; // 从探测器晶体外表面中心到V方向晶体边缘的向量
};

inline CrystalGeometry calculateCrystalGeometry(
    const Coordinate3D<float> &__detector, const DetectorGeometry &__geometry, unsigned __localCrystalIndex) {
  return calculateCrystalGeometry(__detector, __geometry,
                                  __localCrystalIndex % (__geometry.crystalNumU * __geometry.blockNumU),
                                  __localCrystalIndex / (__geometry.crystalNumU * __geometry.blockNumU));
}

inline CrystalGeometry calculateBlockGeometry(
    const Coordinate3D<float> &__detector, const DetectorGeometry &__geometry, uint16_t __blockIndexU,
    uint16_t __blockIndexV) {
  CrystalGeometry result;

  const auto directionU = __detector.rotation.normalized();
  const auto directionNormal = __detector.direction.normalized();
  const auto directionV = directionNormal.cross(directionU).normalized();

  const float blockCenterU =
      (__geometry.blockNumU / -2.f + 0.5f) * __geometry.blockSizeU + __blockIndexU * __geometry.blockSizeU;
  const float blockCenterV =
      (__geometry.blockNumV / -2.f + 0.5f) * __geometry.blockSizeV + __blockIndexV * __geometry.blockSizeV;

  result.position = core::make_vec3<float>(__detector.position + directionU * blockCenterU + directionV * blockCenterV);
  result.directionU = directionU * (__geometry.crystalNumU * __geometry.crystalSizeU) / 2;
  result.directionV = directionV * (__geometry.crystalNumV * __geometry.crystalSizeV) / 2;

  return result;
}

// inline p3df polylinePosition(
//     p3df A, p3df middle, p3df B, float distanceFromA) {
//   const auto A_middle = (A - middle).l2();
//   if (distanceFromA <= A_middle) { // Thus: point is at line segment A-middle
//     return A + (middle - A).normalized() * distanceFromA;
//   } else { // Thus: point is at line segment middle-B
//     const auto distanceFromMiddle = distanceFromA - A_middle;
//     return middle + (B - middle).normalized() * distanceFromMiddle;
//   }
// }
__PNI_CUDA_MACRO__ inline p3df polylinePosition(
    p3df A, p3df middle, p3df B, float distancePA_PB) {
  const auto mA_mB = (A - middle).l2() - (B - middle).l2();
  if (distancePA_PB < mA_mB) { // Thus: point is at line segment A-middle
    const auto PA_PM = distancePA_PB + (B - middle).l2();
    const auto PA = ((A - middle).l2() + PA_PM) / 2;
    return A + (middle - A).normalized() * PA;
  } else { // Thus: point is at line segment middle-B
    const auto PM_PB = distancePA_PB - (A - middle).l2();
    const auto PM = ((B - middle).l2() + PM_PB) / 2;
    return middle + (B - middle).normalized() * PM;
  }
}
} // namespace openpni::experimental::core
