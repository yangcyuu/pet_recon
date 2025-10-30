#pragma once
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

#include "../basic/CudaPtr.hpp"
#include "../basic/Image.hpp"
namespace openpni::process {
struct _JoinImages {
  template <typename ImageValueType, typename OverlapPolicy, typename JoinPolicy>
  inline void operator()(
      const std::vector<ImageValueType const *> &__d_in_ptr,
      const std::vector<basic::Image3DGeometry> &__in_geometry, ImageValueType *__d_out,
      basic::Image3DGeometry __out_geometry, OverlapPolicy __overlap_policy,
      JoinPolicy __join_policy) const {
    if (__d_in_ptr.empty())
      return; // Do nothing
    constexpr int maxImageNum = 64;
    if (__d_in_ptr.size() > maxImageNum)
      throw std::runtime_error("Too many input images for joinImages, max support " +
                               std::to_string(maxImageNum) + " images.");
    auto _d_in_ptrs = make_cuda_sync_ptr_from_hcopy(__d_in_ptr);
    auto _d_in_geometry = make_cuda_sync_ptr_from_hcopy(__in_geometry);
    const auto d_in_ptr = _d_in_ptrs.get();
    const auto d_in_geometry = _d_in_geometry.get();
    const auto imageNum = __d_in_ptr.size();
    thrust::for_each(thrust::counting_iterator<size_t>(0),
                     thrust::counting_iterator<size_t>(__out_geometry.totalVoxelNum()),
                     [=] __device__(size_t idx) {
                       const auto [X, Y, Z] = __out_geometry.at(idx);
                       const auto voxelCenter = __out_geometry.voxelCenter(X, Y, Z);
                       ImageValueType overlapValue[maxImageNum];
                       ImageValueType imageValue[maxImageNum];
                       ImageValueType overlapValueSum = 0;
                       ImageValueType resultValue = 0;
                       for (int i = 0; i < imageNum; ++i)
                         overlapValueSum += overlapValue[i] =
                             __overlap_policy.percent(d_in_geometry[i], voxelCenter);
                       if (overlapValueSum < 1e-8)
                         return; // Dont change the origin value.
                       for (int i = 0; i < imageNum; ++i)
                         overlapValue[i] /= overlapValueSum;
                       for (int i = 0; i < imageNum; i++)
                         imageValue[i] = __overlap_policy.imageValue(
                             d_in_geometry[i], voxelCenter, d_in_ptr[i]);
                       resultValue = __join_policy(overlapValue, imageValue, imageNum);
                       __d_out[idx] = resultValue;
                     });
  }

  struct JoinPolicyAverage {
    template <typename ImageValueType>
    __PNI_CUDA_MACRO__ ImageValueType operator()(
        const ImageValueType *overlapValue, const ImageValueType *imageValue,
        int imageNum) const {
      ImageValueType weightedSum = 0;
      for (int i = 0; i < imageNum; ++i)
        weightedSum += overlapValue[i] * imageValue[i];
      return weightedSum;
    }
  };
  struct JoinPolicyLastAlways {
    template <typename ImageValueType>
    __PNI_CUDA_MACRO__ ImageValueType operator()(
        const ImageValueType *overlapValue, const ImageValueType *imageValue,
        int imageNum) const {
      return imageValue[imageNum - 1];
    }
  };
  struct OverlapPolicyNearest {
    template <typename ImageValueType>
    __PNI_CUDA_MACRO__ ImageValueType imageValue(
        const basic::Image3DGeometry &geometry, const basic::Vec3<float> &voxelCoord,
        const ImageValueType *imageData) const {
      const auto coordInt = basic::make_vec3<int>(voxelCoord);
      if (geometry.in(coordInt))
        return imageData[geometry.at(coordInt)];
      else
        return 0;
    }
    template <typename ImageValueType>
    __PNI_CUDA_MACRO__ float percent(
        const basic::Image3DGeometry &geometry,
        const basic::Vec3<float> &voxelCoord) const {
      const auto coordInt = basic::make_vec3<int>(voxelCoord);
      if (geometry.in(coordInt))
        return 1;
      else
        return 0;
    }
  };
  struct OverlapPolicyTrilinear {
    template <typename ImageValueType>
    __PNI_CUDA_MACRO__ ImageValueType imageValue(
        const basic::Image3DGeometry &geometry, const basic::Vec3<float> &voxelCoord,
        const ImageValueType *imageData) const {
      const auto index = (voxelCoord - geometry.imgBegin) / geometry.voxelSize - 0.5;
      const auto i000 = basic::make_vec3<int>(basic::point_floor(index));

      const basic::Vec3<float> cubeBegin =
          geometry.imgBegin + geometry.voxelSize.pointWiseMul(i000 + float(0.5));
      const basic::Vec3<float> cubeEnd =
          geometry.imgBegin + geometry.voxelSize.pointWiseMul(
                                  (i000 + basic::make_vec3<int>(1, 1, 1) + float(0.5)));
      const basic::Vec3<float> uniformedPoint =
          (voxelCoord - cubeBegin) / (cubeEnd - cubeBegin);

      const auto i100 = i000 + basic::make_vec3<int>(1, 0, 0);
      const auto i010 = i000 + basic::make_vec3<int>(0, 1, 0);
      const auto i110 = i000 + basic::make_vec3<int>(1, 1, 0);
      const auto i001 = i000 + basic::make_vec3<int>(0, 0, 1);
      const auto i101 = i000 + basic::make_vec3<int>(1, 0, 1);
      const auto i011 = i000 + basic::make_vec3<int>(0, 1, 1);
      const auto i111 = i000 + basic::make_vec3<int>(1, 1, 1);

      float _000 = geometry.in(i000) ? imageData[geometry.at(i000)] : 0;
      float _100 = geometry.in(i100) ? imageData[geometry.at(i100)] : 0;
      float _010 = geometry.in(i010) ? imageData[geometry.at(i010)] : 0;
      float _110 = geometry.in(i110) ? imageData[geometry.at(i110)] : 0;
      float _001 = geometry.in(i001) ? imageData[geometry.at(i001)] : 0;
      float _101 = geometry.in(i101) ? imageData[geometry.at(i101)] : 0;
      float _011 = geometry.in(i011) ? imageData[geometry.at(i011)] : 0;
      float _111 = geometry.in(i111) ? imageData[geometry.at(i111)] : 0;

      _000 *= (1 - uniformedPoint.x) * (1 - uniformedPoint.y) * (1 - uniformedPoint.z);
      _100 *= uniformedPoint.x * (1 - uniformedPoint.y) * (1 - uniformedPoint.z);
      _010 *= (1 - uniformedPoint.x) * uniformedPoint.y * (1 - uniformedPoint.z);
      _110 *= uniformedPoint.x * uniformedPoint.y * (1 - uniformedPoint.z);
      _001 *= (1 - uniformedPoint.x) * (1 - uniformedPoint.y) * uniformedPoint.z;
      _101 *= uniformedPoint.x * (1 - uniformedPoint.y) * uniformedPoint.z;
      _011 *= (1 - uniformedPoint.x) * uniformedPoint.y * uniformedPoint.z;
      _111 *= uniformedPoint.x * uniformedPoint.y * uniformedPoint.z;

      return _000 + _100 + _010 + _110 + _001 + _101 + _011 + _111;
    }
    template <typename ImageValueType>
    __PNI_CUDA_MACRO__ float percent(
        const basic::Image3DGeometry &geometry,
        const basic::Vec3<float> &voxelCoord) const {
      const auto index = (voxelCoord - geometry.imgBegin) / geometry.voxelSize - 0.5;
      const auto i000 = basic::make_vec3<int>(basic::point_floor(index));

      const basic::Vec3<float> cubeBegin =
          geometry.imgBegin + geometry.voxelSize.pointWiseMul(i000 + float(0.5));
      const basic::Vec3<float> cubeEnd =
          geometry.imgBegin + geometry.voxelSize.pointWiseMul(
                                  (i000 + basic::make_vec3<int>(1, 1, 1) + float(0.5)));
      const basic::Vec3<float> uniformedPoint =
          (voxelCoord - cubeBegin) / (cubeEnd - cubeBegin);

      const auto i100 = i000 + basic::make_vec3<int>(1, 0, 0);
      const auto i010 = i000 + basic::make_vec3<int>(0, 1, 0);
      const auto i110 = i000 + basic::make_vec3<int>(1, 1, 0);
      const auto i001 = i000 + basic::make_vec3<int>(0, 0, 1);
      const auto i101 = i000 + basic::make_vec3<int>(1, 0, 1);
      const auto i011 = i000 + basic::make_vec3<int>(0, 1, 1);
      const auto i111 = i000 + basic::make_vec3<int>(1, 1, 1);

      float _000 = geometry.in(i000) ? 1 : 0;
      float _100 = geometry.in(i100) ? 1 : 0;
      float _010 = geometry.in(i010) ? 1 : 0;
      float _110 = geometry.in(i110) ? 1 : 0;
      float _001 = geometry.in(i001) ? 1 : 0;
      float _101 = geometry.in(i101) ? 1 : 0;
      float _011 = geometry.in(i011) ? 1 : 0;
      float _111 = geometry.in(i111) ? 1 : 0;

      _000 *= (1 - uniformedPoint.x) * (1 - uniformedPoint.y) * (1 - uniformedPoint.z);
      _100 *= uniformedPoint.x * (1 - uniformedPoint.y) * (1 - uniformedPoint.z);
      _010 *= (1 - uniformedPoint.x) * uniformedPoint.y * (1 - uniformedPoint.z);
      _110 *= uniformedPoint.x * uniformedPoint.y * (1 - uniformedPoint.z);
      _001 *= (1 - uniformedPoint.x) * (1 - uniformedPoint.y) * uniformedPoint.z;
      _101 *= uniformedPoint.x * (1 - uniformedPoint.y) * uniformedPoint.z;
      _011 *= (1 - uniformedPoint.x) * uniformedPoint.y * uniformedPoint.z;
      _111 *= uniformedPoint.x * uniformedPoint.y * uniformedPoint.z;

      return _000 + _100 + _010 + _110 + _001 + _101 + _011 + _111;
    }
  };
};
inline constexpr _JoinImages joinImages = {};
} // namespace openpni::process