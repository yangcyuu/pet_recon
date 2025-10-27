#pragma once
#include <cinttypes>
#include <numbers>

#include "../tools/Consts.hpp"
#include "Vector.hpp"
namespace openpni::experimental::core {
template <FloatingPoint_c T>
using cube = experimental::core::Vector<experimental::core::Vector<T, 3>, 2>;

template <FloatingPoint_c F, int D>
struct DirectionalLineSegment { // 有向线段
  Vector<F, D> O;               // 起点坐标
  Vector<F, D> OP;              // 起点到终点的向量
  __PNI_CUDA_MACRO__ static auto create_by_two_points(
      const Vector<F, D> &origin, const Vector<F, D> &target) {
    DirectionalLineSegment<F, D> result;
    result.O = origin;
    result.OP = target - origin;
    return result;
  }
  __PNI_CUDA_MACRO__ static auto create_by_two_points_and_range(
      const Vector<F, D> &origin, const Vector<F, D> &target, F amin, F amax) {
    DirectionalLineSegment<F, D> temp = create_by_two_points(origin, target);
    DirectionalLineSegment<F, D> result;
    result.O = temp.getPoint(amin);
    auto p = temp.getPoint(amax);
    result.OP = p - result.O;
    return result;
  }
  __PNI_CUDA_MACRO__ Vector<F, D> getPoint(
      // From 0<=>origin to 1<=>target
      F t) const {
    return O + OP * t;
  }
  __PNI_CUDA_MACRO__ F getLength() const { return algorithms::l2(OP); }
};

template <FloatingPoint_c F, int D>
struct Triangle {  // 三角形，可二维可三维
  Vector<F, D> O;  // 顶点O坐标
  Vector<F, D> OA; // 顶点O到顶点A的向量
  Vector<F, D> OB; // 顶点O到顶点B的向量
  __PNI_CUDA_MACRO__ static auto create_by_three_points(
      const Vector<F, D> &O, const Vector<F, D> &A, const Vector<F, D> &B) {
    Triangle<F, D> result;
    result.O = O;
    result.OA = A - O;
    result.OB = B - O;
    return result;
  }
  __PNI_CUDA_MACRO__ Vector<F, D> getPoint(
      F u, F v) const {
    return O + OA * u + OB * v;
  }
};

template <FloatingPoint_c T>
__PNI_CUDA_MACRO__ cube<T> cube_absolute(
    const cube<T> &__roi) // 将立方体范围转换成相同但正规的表达
{
  using fmath = FMath<T>;
  auto minVec = Vector<T, 3>::create(fmath::min(__roi[0][0], __roi[1][0]), fmath::min(__roi[0][1], __roi[1][1]),
                                     fmath::min(__roi[0][2], __roi[1][2]));
  auto maxVec = Vector<T, 3>::create(fmath::max(__roi[0][0], __roi[1][0]), fmath::max(__roi[0][1], __roi[1][1]),
                                     fmath::max(__roi[0][2], __roi[1][2]));
  return Vector<Vector<T, 3>, 2>::create(minVec, maxVec);
}
template <FloatingPoint_c T>
__PNI_CUDA_MACRO__ cube<T> cube_and(
    // 计算两个立方体的交集
    const cube<T> &__roi1, // => roi_min
    const cube<T> &__roi2  // => roi_max
) {
  using fmath = FMath<T>;
  return Vector<Vector<T, 3>, 2>::create(
      Vector<T, 3>::create(fmath::max(__roi1[0][0], __roi2[0][0]), fmath::max(__roi1[0][1], __roi2[0][1]),
                           fmath::max(__roi1[0][2], __roi2[0][2])),
      Vector<T, 3>::create(fmath::min(__roi1[1][0], __roi2[1][0]), fmath::min(__roi1[1][1], __roi2[1][1]),
                           fmath::min(__roi1[1][2], __roi2[1][2])));
}

template <FloatingPoint_c F>
struct RectangleGeom { // 这个结构体用来描述3D空间中的一个平面矩形
  Vector<F, 3> O;      // 中心坐标
  Vector<F, 3> U;      // u方向向量，长度为半U宽或1（取决于具体调用）
  Vector<F, 3> V;      // v方向向量，长度为半V宽或1（取决于具体调用）
  // 要求：U×V方向（法线）是朝向远离探测器的方向
};

struct DetectorGeom {
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
  uint16_t getTotalCrystalU() const { return blockNumU * crystalNumU; }
  uint16_t getTotalCrystalV() const { return blockNumV * crystalNumV; }
  uint16_t getTotalCrystalNum() const { return getTotalBlockNum() * getCrystalNumInBlock(); }
  float getUSize() const { return blockNumU * blockSizeU; }
  float getVSize() const { return blockNumV * blockSizeV; }
  uint16_t getUBlkIdx(
      int uCryIdx) const {
    return uCryIdx / crystalNumU;
  }
  uint16_t getVBlkIdx(
      int uCryIdx) const {
    return uCryIdx / crystalNumU;
  }
  float getUBlkCenter(
      int uBlkIdx) const {
    return (-blockNumU / 2.f + 0.5f + uBlkIdx) * blockSizeU;
  }
  float getVBlkCenter(
      int vBlkIdx) const {
    return (-blockNumV / 2.f + 0.5f + vBlkIdx) * blockSizeV;
  }
  float getUCryCenter(
      int uCryIdx) const {
    return getUBlkCenter(getUBlkIdx(uCryIdx)) + (-crystalNumU / 2.f + 0.5f + (uCryIdx % crystalNumU)) * crystalSizeU;
  }
  float getVCryCenter(
      int vCryIdx) const {
    return getVBlkCenter(getVBlkIdx(vCryIdx)) + (-crystalNumV / 2.f + 0.5f + (vCryIdx % crystalNumV)) * crystalSizeV;
  }
};

using CrystalGeom = RectangleGeom<float>;
using DetectorCoord = RectangleGeom<float>;

struct PolygonalSystem // 多边形系统，也即同构环系统
{
  unsigned edges;           // 多边形的边数
  unsigned detectorPerEdge; // 每条边有几个探测器
  unsigned detectorLen;     // 探测器中心距离
  float radius;             // 中心到边的距离
  float angleOf1stPerp;     // 第一条垂线对应的角度
  unsigned detectorRings;   // 探测器组成的环数
  float ringDistance;       // 相邻两环之间的中心距离
  unsigned hyperRingNum;    // 超环数
  float hyperRingDistance;  // 超环之间的距离

  unsigned getTotalDetectorNum() const { return edges * detectorPerEdge * detectorRings; }
  using DetectorGlobalCoor = DetectorCoord;
  DetectorGlobalCoor getDetectorGlobalCoor(
      unsigned __detectorRingIndex, unsigned __detectorIndex) const {
    using V = core::Vector<float, 3>;

    const float edgeLenth = detectorPerEdge * detectorLen;
    const float rotateAngle = 2.0f * std::numbers::pi_v<float> / edges; // 每条边的旋转角度
    // 计算detectorIndex所在的边和位置
    const unsigned edgeIndex = __detectorIndex / detectorPerEdge;
    // 计算该edge的中心
    const float angleNow = angleOf1stPerp + edgeIndex * rotateAngle;

    const auto midP = V::create( // 从X轴正方向开始逆时针时针旋转
                          std::cos(angleNow), std::sin(angleNow), 0) *
                      float(radius);
    // 计算detector在该边上的位置0
    const unsigned positionIndex = __detectorIndex % detectorPerEdge;

    auto v = V::create(std::cos(angleNow + tools::pi_2), std::sin(angleNow + tools::pi_2), 0);
    // const auto p0 = midP - v * (edgeLenth / 2.0f);
    const auto p0 = midP + v * detectorLen * (detectorPerEdge / -2.f + .5f);
    // 根据端点求出第一个探测器的坐标
    // const auto dete0 = p0 + v * (__polygon.detectorLen / 2.0f);
    // const auto deteNow = dete0 + v * (__polygon.detectorLen * positionIndex);
    const auto p1 = p0 + v * detectorLen * positionIndex;
    const auto p2 = p1 + V::create(0, 0, 1) * (detectorRings / -2.f + .5f) * ringDistance;
    const auto p3 = p2 + V::create(0, 0, 1) * ringDistance * __detectorRingIndex;

    DetectorGlobalCoor coord;
    coord.O = p3;
    coord.U = V::create(0.0f, 0.0f, 1.0f); // U方向为z轴正方向
    coord.V = algorithms::normalized(-v);  // V方向为指向顺时针
    return coord;
  }
  DetectorGlobalCoor getDetectorGlobalCoor(
      unsigned __detectorGlobalIndex) const {
    const unsigned detectorsPerRing = edges * detectorPerEdge;
    const unsigned ringIndex = __detectorGlobalIndex / detectorsPerRing;
    const unsigned indexInRing = __detectorGlobalIndex % detectorsPerRing;
    return getDetectorGlobalCoor(ringIndex, indexInRing);
  }
};

inline CrystalGeom calculateCrystalGeometry(
    const DetectorCoord &__detector, const DetectorGeom &__geometry, uint16_t __crystalIndexU,
    uint16_t __crystalIndexV) {
  CrystalGeom result;

  const auto directionU = algorithms::normalized(__detector.U);
  const auto directionV = algorithms::normalized(__detector.V);

  const auto blockIndexU = __crystalIndexU / __geometry.crystalNumU;
  const auto blockIndexV = __crystalIndexV / __geometry.crystalNumV;
  const auto crystalIndexU = __crystalIndexU % __geometry.crystalNumU;
  const auto crystalIndexV = __crystalIndexV % __geometry.crystalNumV;

  const float blockCenterU =
      (__geometry.blockNumU / -2.f + 0.5f) * __geometry.blockSizeU + blockIndexU * __geometry.blockSizeU;
  const float blockCenterV =
      (__geometry.blockNumV / -2.f + 0.5f) * __geometry.blockSizeV + blockIndexV * __geometry.blockSizeV;
  const float crystalCenterU =
      (__geometry.crystalNumU / -2.f + 0.5f) * __geometry.crystalSizeU + crystalIndexU * __geometry.crystalSizeU;
  const float crystalCenterV =
      (__geometry.crystalNumV / -2.f + 0.5f) * __geometry.crystalSizeV + crystalIndexV * __geometry.crystalSizeV;

  result.O =
      (__detector.O + directionU * (blockCenterU + crystalCenterU) + directionV * (blockCenterV + crystalCenterV))
          .template to<float>();
  result.U = directionU * __geometry.crystalSizeU / 2;
  result.V = directionV * __geometry.crystalSizeV / 2;

  return result;
}

} // namespace openpni::experimental::core

namespace openpni::experimental::algorithms {}