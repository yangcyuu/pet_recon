#pragma once
#include <memory>
#include <ranges>
#include <vector>

#include "../experimental/core/BasicMath.hpp"
#include "Math.hpp"
#include "Point.hpp"
namespace openpni::basic {
struct Image3DGeometry {
  Vec3<float> voxelSize;
  Vec3<float> imgBegin;
  Vec3<int> voxelNum;

  __PNI_CUDA_MACRO__ auto centre() const { return imgBegin + voxelSize.pointWiseMul(voxelNum) / 2; }

  __PNI_CUDA_MACRO__ std::size_t totalVoxelNum() const {
    return std::size_t(voxelNum.x) * std::size_t(voxelNum.y) * std::size_t(voxelNum.z);
  }

  template <Integral_c T>
  __PNI_CUDA_MACRO__ std::size_t at(
      const Vec3<T> &__index) const {
    return at(static_cast<int>(__index.x), static_cast<int>(__index.y), static_cast<int>(__index.z));
  }
  __PNI_CUDA_MACRO__ std::size_t at(
      int __x, int __y, int __z) const {
    return __x + voxelNum.x * (__y + voxelNum.y * (__z));
  }
  __PNI_CUDA_MACRO__ Vec3<int> at(
      std::size_t idx) const {
    int z = static_cast<int>(idx / (std::size_t(voxelNum.x) * std::size_t(voxelNum.y)));
    idx %= (std::size_t(voxelNum.x) * std::size_t(voxelNum.y));
    int y = static_cast<int>(idx / std::size_t(voxelNum.x));
    int x = static_cast<int>(idx % std::size_t(voxelNum.x));
    return Vec3<int>{x, y, z};
  }
  __PNI_CUDA_MACRO__ bool in(
      int __x, int __y, int __z) const {
    return __x < voxelNum.x && __y < voxelNum.y && __z < voxelNum.z && __x >= 0 && __y >= 0 && __z >= 0;
  }

  template <Integral_c T>
  __PNI_CUDA_MACRO__ bool in(
      const Vec3<T> &__p) const {
    return in(__p.x, __p.y, __p.z);
  }

  __PNI_CUDA_MACRO__ cubef roi() const {
    return make_vec2<Vec3<float>>(imgBegin, imgBegin + voxelSize.pointWiseMul(voxelNum));
  }

  template <Integral_c T>
  __PNI_CUDA_MACRO__ cubef voxel_roi(
      const Vec3<T> &__p) const {
    return make_vec2<Vec3<float>>(imgBegin + voxelSize.pointWiseMul(__p), imgBegin + voxelSize.pointWiseMul(__p + 1));
  }
  template <Integral_c T>
  __PNI_CUDA_MACRO__ p3df voxel_center(
      const Vec3<T> &__p) const {
    return imgBegin + voxelSize.pointWiseMul(__p + 0.5f);
  }

  std::vector<p3df> voxel_centers() const {
    std::vector<p3df> pts;
    pts.reserve(totalVoxelNum());
    for (unsigned z = 0; z < voxelNum.z; ++z)
      for (unsigned y = 0; y < voxelNum.y; ++y)
        for (unsigned x = 0; x < voxelNum.x; ++x)
          pts.push_back(voxel_center(Vec3<unsigned>(x, y, z)));
    return pts;
  }
};

__PNI_CUDA_MACRO__ inline Image3DGeometry make_ImageSize(
    const Vec3<float> &__voxelSize, const Vec3<float> &__imgBegin, const Vec3<int> &__voxelDim) {
  Image3DGeometry s;
  s.voxelSize = __voxelSize;
  s.imgBegin = __imgBegin;
  s.voxelNum = __voxelDim;
  return s;
}

__PNI_CUDA_MACRO__ inline Image3DGeometry make_ImageSizeByCenter(
    const Vec3<float> &__voxelSize, const Vec3<float> &__imgCenter, const Vec3<int> &__voxelDim) {
  return make_ImageSize(__voxelSize, __imgCenter - __voxelSize.pointWiseMul(__voxelDim) / 2, __voxelDim);
}

struct Image2DGeometry {
  Vec2<float> voxelSize;
  Vec2<float> imgBegin;
  Vec2<unsigned> voxelNum;

  __PNI_CUDA_MACRO__ std::size_t totalVoxelNum() const { return std::size_t(voxelNum.x) * std::size_t(voxelNum.y); }

  template <Integral_c T>
  __PNI_CUDA_MACRO__ std::size_t at(
      const Vec2<T> &__p) const {
    return at(__p.x, __p.y);
  }

  __PNI_CUDA_MACRO__ std::size_t at(
      unsigned __x, unsigned __y) const {
    return __x + voxelNum.x * (__y);
  }

  __PNI_CUDA_MACRO__ bool in(
      unsigned __x, unsigned __y) const {
    return __x < voxelNum.x && __y < voxelNum.y;
  }
  template <Integral_c T>
  __PNI_CUDA_MACRO__ bool in(
      const Vec2<T> &__p) const {
    return in(__p.x, __p.y);
  }
  template <Integral_c T>
  __PNI_CUDA_MACRO__ Vec2<float> voxel_center(
      const Vec2<T> &__p) const {
    return imgBegin + voxelSize.pointWiseMul(__p + 0.5f);
  }
};

__PNI_CUDA_MACRO__ inline Image2DGeometry make_ImageSize(
    const Vec2<float> &__voxelSize, const Vec2<float> &__imgBegin, const Vec2<unsigned> &__voxelDim) {
  Image2DGeometry s;
  s.voxelSize = __voxelSize;
  s.imgBegin = __imgBegin;
  s.voxelNum = __voxelDim;
  return s;
}
__PNI_CUDA_MACRO__ inline Image2DGeometry make_ImageSizeByCenter(
    const Vec2<float> &__voxelSize, const Vec2<float> &__imgCenter, const Vec2<unsigned> &__voxelDim) {
  return make_ImageSize(__voxelSize, __imgCenter - __voxelSize.pointWiseMul(__voxelDim) / 2, __voxelDim);
}

} // namespace openpni::basic

namespace openpni {
template <typename ImageValueType>
struct Image3DSpan {
  basic::Image3DGeometry geometry;
  ImageValueType *ptr;
};
template <typename ImageValueType>
using Image3DInputSpan = Image3DSpan<const ImageValueType>;
template <typename ImageValueType>
using Image3DOutputSpan = Image3DSpan<std::remove_const_t<ImageValueType>>;
template <typename ImageValueType>
struct Image3DIOSpan {
  basic::Image3DGeometry geometry;
  ImageValueType const *ptr_in;
  std::remove_const_t<ImageValueType> *ptr_out;
  auto input_span() const { return Image3DInputSpan<ImageValueType>{geometry, ptr_in}; }
  auto output_span() const { return Image3DOutputSpan<ImageValueType>{geometry, ptr_out}; }
};
} // namespace openpni

namespace openpni::basic {
template <typename Value>
struct Image3DAccessor {
  Image3DAccessor(
      Vec3<int> _voxelNum, Value *_ptr)
      : voxelNum(_voxelNum)
      , ptr(_ptr) {}

  __PNI_CUDA_MACRO__ Value &operator()(
      const Vec3<int> &pos) const {
    return ptr[at(pos)];
  }
  __PNI_CUDA_MACRO__ Value &operator()(
      int x, int y, int z) const {
    return (*this)(Vec3<int>(x, y, z));
  }

  __PNI_CUDA_MACRO__ std::size_t at(
      int x, int y, int z) const {
    return x + voxelNum.x * (y + voxelNum.y * z);
  }
  __PNI_CUDA_MACRO__ std::size_t at(
      const Vec3<int> &pos) const {
    return at(pos.x, pos.y, pos.z);
  }
  __PNI_CUDA_MACRO__ bool in(
      int x, int y, int z) const {
    return x >= 0 && x < voxelNum.x && y >= 0 && y < voxelNum.y && z >= 0 && z < voxelNum.z;
  }
  __PNI_CUDA_MACRO__ bool in(
      const Vec3<int> &pos) const {
    return in(pos.x, pos.y, pos.z);
  }

private:
  Vec3<int> voxelNum;
  Value *ptr;
};

template <typename Value, int CutNum>
struct Image3DIndirectAccessor {
  static_assert(CutNum >= 2, "CutNum must be reasonable");
  static_assert(CutNum < 10, "CutNum must be reasonable");

  Image3DIndirectAccessor(
      Vec3<int> _voxelNum, Value *_ptr)
      : voxelNum(_voxelNum)
      , ptr(_ptr) {
    for (const auto sectorIndex : std::views::iota(0, CutNum * CutNum * CutNum)) {
      const int sector_x = sectorIndex % CutNum;
      const int sector_y = (sectorIndex / CutNum) % CutNum;
      const int sector_z = sectorIndex / (CutNum * CutNum);
      const int size_x = sector_x == CutNum - 1 ? (voxelNum.x - sector_x * (voxelNum.x / CutNum)) : voxelNum.x / CutNum;
      const int size_y = sector_y == CutNum - 1 ? (voxelNum.y - sector_y * (voxelNum.y / CutNum)) : voxelNum.y / CutNum;
      const int size_z = sector_z == CutNum - 1 ? (voxelNum.z - sector_z * (voxelNum.z / CutNum)) : voxelNum.z / CutNum;
      sectorSize[sectorIndex] = Vec3<int>(size_x, size_y, size_z);
    }
    sectorOffset[0] = 0;
    for (const auto sectorIndex : std::views::iota(1, CutNum * CutNum * CutNum)) {
      sectorOffset[sectorIndex] = sectorOffset[sectorIndex - 1] + std::size_t(sectorSize[sectorIndex - 1].x) *
                                                                      std::size_t(sectorSize[sectorIndex - 1].y) *
                                                                      std::size_t(sectorSize[sectorIndex - 1].z);
    }
  }

  __PNI_CUDA_MACRO__ Value &operator()(
      const Vec3<int> &pos) const {
    return ptr[at(pos)];
  }
  __PNI_CUDA_MACRO__ Value &operator()(
      int x, int y, int z) const {
    return (*this)(Vec3<int>(x, y, z));
  }
  __PNI_CUDA_MACRO__ std::size_t at(
      int x, int y, int z) const {
    const int sectorIndexX = x / (voxelNum.x / CutNum);
    const int sectorIndexY = y / (voxelNum.y / CutNum);
    const int sectorIndexZ = z / (voxelNum.z / CutNum);
    const int sectorIndex = sectorIndexX + sectorIndexY * CutNum + sectorIndexZ * CutNum * CutNum;
    return sectorOffset[sectorIndex] + (x - sectorIndexX * (voxelNum.x / CutNum)) +
           (y - sectorIndexY * (voxelNum.y / CutNum)) * sectorSize[sectorIndex].x +
           (z - sectorIndexZ * (voxelNum.z / CutNum)) * sectorSize[sectorIndex].x * sectorSize[sectorIndex].y;
  }
  __PNI_CUDA_MACRO__ std::size_t at(
      const Vec3<int> &pos) const {
    return at(pos.x, pos.y, pos.z);
  }
  __PNI_CUDA_MACRO__ bool in(
      int x, int y, int z) const {
    return x >= 0 && x < voxelNum.x && y >= 0 && y < voxelNum.y && z >= 0 && z < voxelNum.z;
  }
  __PNI_CUDA_MACRO__ bool in(
      const Vec3<int> &pos) const {
    return in(pos.x, pos.y, pos.z);
  }

private:
  Vec3<int> voxelNum;
  std::size_t sectorOffset[CutNum * CutNum * CutNum];
  Vec3<int> sectorSize[CutNum * CutNum * CutNum];
  Value *ptr;
};
} // namespace openpni::basic
