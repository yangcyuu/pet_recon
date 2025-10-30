#pragma once
#include "../core/BasicMath.hpp"
#include "../core/Geometric.hpp"
namespace openpni::experimental::algorithms {
template <FloatingPoint_c T>
__PNI_CUDA_MACRO__ core::Vector<T, 2> liang_barskey_3d(
    core::cube<T> const &roi, core::Vector<T, 3> const &p1, core::Vector<T, 3> const &p2) {
  using fmath = core::FMath<T>;
  const auto &[cubeOrigin, cubeEnd] = roi;
  const auto p2_p1 = p2 - p1;

  const T ax0 = (cubeOrigin[0] - p1[0]) / p2_p1[0];
  const T axn = (cubeEnd[0] - p1[0]) / p2_p1[0];
  const T axmin = fmath::min(ax0, axn);
  const T axmax = fmath::max(ax0, axn);

  const T ay0 = (cubeOrigin[1] - p1[1]) / p2_p1[1];
  const T ayn = (cubeEnd[1] - p1[1]) / p2_p1[1];
  const T aymin = fmath::min(ay0, ayn);
  const T aymax = fmath::max(ay0, ayn);

  const T az0 = (cubeOrigin[2] - p1[2]) / p2_p1[2];
  const T azn = (cubeEnd[2] - p1[2]) / p2_p1[2];
  const T azmin = fmath::min(az0, azn);
  const T azmax = fmath::max(az0, azn);

  const T amax = fmath::min(fmath::min(T(1), axmax), fmath::min(aymax, azmax));
  const T amin = fmath::max(fmath::max(T(0), axmin), fmath::max(aymin, azmin));

  return core::Vector<T, 2>::create(amin, amax);
}
} // namespace openpni::experimental::algorithms
