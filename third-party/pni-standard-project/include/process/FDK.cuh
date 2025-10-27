#pragma once
#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <memory>
#include <stdio.h>
#include <thread>
#include <thrust/device_ptr.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

#include "../basic/Image.hpp"
#include "../basic/Math.hpp"
#include "../math/Convolution.hpp"
#include "../math/FouriorFilter.cuh"
#include "../math/Geometry.hpp"
#include "../math/Interpolation.hpp"
#include "../math/Transforms.hpp"
#include "../process/Foreach.cuh"
#include "Foreach.cuh"
#include "Foreach.hpp"
namespace openpni::process {
struct CBCTProjectionInfo {
  basic::Vec3<float> positionX;  // X射线源的3D位置
  basic::Vec3<float> directionU; // 探测器平面U方向的单位向量
  basic::Vec3<float> directionV; // 探测器平面V方向的单位向量
  basic::Vec3<float> positionD;  // 探测器平面上中心点的3D位置
};
template <typename _ProjectionValueType>
struct CBCTDataView {
  using ProjectionValueType = _ProjectionValueType;
  CBCTProjectionInfo *projectionInfo;
  ProjectionValueType const *const *projectionDataPtrs; // 投影图值，一级指针在内存，数据在内存或GPU上
  ProjectionValueType const *const *airDataPtrs;        // 空气扫描值，一级指针在内存，数据在内存或GPU上
  int projectionNum;                                    // 有多少张投影图
  basic::Vec2<unsigned> pixels;                         // 投影图像素数量
  basic::Vec2<float> pixelSize;                         // 投影图每个像素的大小
  float geo_angle;                                      // 几何校正里面的角度
  float geo_offsetU;                                    // 几何校正里面的U方向偏移
  float geo_offsetV;                                    // 几何校正里面的V方向偏移
  float geo_SDD;                                        // 几何校正里面的源到探测器距离
  float geo_SOD;                                        // 几何校正里面的源到中心点距离
  unsigned fouriorCutoffLength;                         // 傅立叶平滑核的截止长度
  float beamHardenParamA;                               // 束硬化校正参数A
  float beamHardenParamB;                               // 束硬化校正参数B
};

template <FloatingPoint_c Precision, typename InterpolationMethod>
inline __PNI_CUDA_MACRO__ Precision fdkProjectionNew_impl(
    Precision const *__projections, const CBCTProjectionInfo *__projectionInfo, int __projectionNum,
    basic::Image3DGeometry __img3dGeometry, basic::Image2DGeometry __img2dGeometry, InterpolationMethod __ipMethod,
    unsigned __X, unsigned __Y, unsigned __Z) {
  Precision sum = 0;
  const auto voxelPosition = __img3dGeometry.voxel_center(basic::make_vec3<unsigned>(__X, __Y, __Z));
  for (int i = 0; i < __projectionNum; i++) {
    const auto &XPosition = __projectionInfo[i].positionX;               // X射线源的3D位置
    const auto UDirection = __projectionInfo[i].directionU.normalized(); // 探测器平面U方向的单位向量
    const auto VDirection = __projectionInfo[i].directionV.normalized(); // 探测器平面V方向的单位向量
    const auto &DPosition = __projectionInfo[i].positionD;               // 探测器平面上中心点的3D位置
    const auto line_from_X_to_voxel = basic::Line<Precision>::create_from_ends(XPosition, voxelPosition);
    const auto plane_of_detector = basic::Plane<Precision>::create(DPosition, UDirection, VDirection);
    const auto intersection = basic::intersection(line_from_X_to_voxel, plane_of_detector);
    const auto intersectionInProjection_wrt_detectorCenter =
        basic::make_vec2<Precision>(intersection.u_plane,
                                    intersection.v_plane); // 交点相对于探测器中心的坐标
    sum += __ipMethod(intersectionInProjection_wrt_detectorCenter, &__projections[i * __img2dGeometry.totalVoxelNum()],
                      __img2dGeometry);
  }
  return sum;
}

struct _FDK_CUDA {
  template <typename ProjectionValueType, FloatingPoint_c CalculatePrecision = float>
  struct air {
    ProjectionValueType const *const *projectionPtrs;
    ProjectionValueType const *const *airValuePtrs;
    std::size_t crystalNum;
    CalculatePrecision *outProjectionValue;
    __host__ __device__ air(
        ProjectionValueType const *const *__projectionPtrs, ProjectionValueType const *const *__airValuePtrs,
        std::size_t __crystalNum, CalculatePrecision *__outProjectionValue)
        : projectionPtrs(__projectionPtrs)
        , airValuePtrs(__airValuePtrs)
        , crystalNum(__crystalNum)
        , outProjectionValue(__outProjectionValue) {}
    __host__ __device__ void operator()(
        std::size_t idx) const {
      using fmath = basic::FMath<CalculatePrecision>;
      const auto projectionId = idx / crystalNum;
      const auto pixelId = idx % crystalNum;
      outProjectionValue[idx] =
          CalculatePrecision(-1.) * fmath::flog(CalculatePrecision(projectionPtrs[projectionId][pixelId]) /
                                                CalculatePrecision(airValuePtrs[projectionId][pixelId]));
    }
  };

  template <FloatingPoint_c CalculatePrecision = float,
            typename InterpolationMethod = math::InterpolationBilinear2D<float>>
  struct geometry {
    CalculatePrecision const *projections_in;
    CalculatePrecision *projections_out;
    basic::Image2DGeometry geometry2D;
    CalculatePrecision offsetU;
    CalculatePrecision offsetV;
    CalculatePrecision angle;
    InterpolationMethod ipMethod;

    process::ActionTranslate2D<CalculatePrecision> actionTranslateProjection;
    process::ActionRotate2D<CalculatePrecision> actionRotateProjection;

    __host__ __device__ geometry(
        CalculatePrecision const *__projections_in, CalculatePrecision *__projections_out,
        basic::Image2DGeometry __geometry, CalculatePrecision __offsetU, CalculatePrecision __offsetV,
        CalculatePrecision __angle, InterpolationMethod __ipMethod)
        : projections_in(__projections_in)
        , projections_out(__projections_out)
        , geometry2D(__geometry)
        , offsetU(__offsetU)
        , offsetV(__offsetV)
        , angle(__angle)
        , ipMethod(__ipMethod)
        , actionTranslateProjection{basic::make_vec2<CalculatePrecision>(-__offsetU, -__offsetV)}
        , actionRotateProjection{basic::make_vec2<CalculatePrecision>((__geometry.voxelNum - 1) / 2.0),
                                 basic::rotationMatrix<CalculatePrecision>(__angle)} {}

    __device__ void operator()(
        std::size_t idx) const {
      const auto projectionId = idx / geometry2D.totalVoxelNum();
      const auto pixelId = idx % geometry2D.totalVoxelNum();
      const auto projectionPtrIn = &projections_in[projectionId * geometry2D.totalVoxelNum()];
      const auto projectionPtrOut = &projections_out[projectionId * geometry2D.totalVoxelNum()];
      const auto x = pixelId % geometry2D.voxelNum.x;
      const auto y = pixelId / geometry2D.voxelNum.x;
      _impl::transform2D_impl(projectionPtrIn, projectionPtrOut, geometry2D, geometry2D, ipMethod, x, y,
                              actionTranslateProjection, actionRotateProjection);
    }
  };

  struct BeamHardenMethods {
    template <FloatingPoint_c CalculatePrecision = float>
    struct v1 {
      CalculatePrecision const *inProjections;
      CalculatePrecision *outProjections;

      __host__ __device__ v1(
          CalculatePrecision const *__inProjections, CalculatePrecision *__outProjections)
          : inProjections(__inProjections)
          , outProjections(__outProjections) {}

      __host__ __device__ void operator()(
          std::size_t idx) const {
        using fmath = basic::FMath<CalculatePrecision>;
        outProjections[idx] = inProjections[idx] < CalculatePrecision(1e-6)
                                  ? CalculatePrecision(0)
                                  : fmath::fpow(fmath::abs(fmath::fexp(inProjections[idx]) - CalculatePrecision(1)),
                                                CalculatePrecision(0.8));
      }
    };
    template <FloatingPoint_c CalculatePrecision = float>
    struct v2 {
      CalculatePrecision paramA;
      CalculatePrecision paramB;
      CalculatePrecision const *inProjections;
      CalculatePrecision *outProjections;

      __host__ __device__ v2(
          CalculatePrecision const *__inProjections, CalculatePrecision *__outProjections, CalculatePrecision __paramA,
          CalculatePrecision __paramB)
          : inProjections(__inProjections)
          , outProjections(__outProjections)
          , paramA(__paramA)
          , paramB(__paramB) {}

      __host__ __device__ void operator()(
          std::size_t idx) const {
        using fmath = basic::FMath<CalculatePrecision>;
        const auto &temp = inProjections[idx];
        outProjections[idx] = temp + paramA * temp * temp + paramB * temp * temp * temp;
      }
    };
  };

  template <FloatingPoint_c CalculatePrecision = float>
  struct weight {
    CalculatePrecision const *projections_in;
    CalculatePrecision *projections_out;
    basic::Image2DGeometry geometry;
    float SDD;

    __host__ __device__ weight(
        CalculatePrecision const *__projections_in, CalculatePrecision *__projections_out,
        basic::Image2DGeometry __geometry, float __SDD)
        : projections_in(__projections_in)
        , projections_out(__projections_out)
        , geometry(__geometry)
        , SDD(__SDD) {}

    __device__ void operator()(
        std::size_t idx) const {
      const auto projectionId = idx / geometry.totalVoxelNum();
      const auto pixelId = idx % geometry.totalVoxelNum();
      const auto projectionPtrIn = &projections_in[projectionId * geometry.totalVoxelNum()];
      const auto projectionPtrOut = &projections_out[projectionId * geometry.totalVoxelNum()];
      const auto indexInProjection =
          basic::make_vec2<unsigned>(pixelId % geometry.voxelNum.x, pixelId / geometry.voxelNum.y);
      const auto positionInProjection = geometry.voxel_center(indexInProjection);
      projectionPtrOut[pixelId] = projectionPtrIn[pixelId] * (SDD / sqrt(positionInProjection.l22() + SDD * SDD));
    }
  };

  template <FloatingPoint_c CalculatePrecision = float>
  inline void fouriorFilterCUDA(
      CalculatePrecision const *__d_projectionPtrs_in, // p = host, *p = host, **p = device
      CalculatePrecision *__d_projectionPtrs_out,      // p = host, *p = host, **p = device
      basic::Image2DGeometry __geometry, int __angleNum, unsigned __fouriorCutoffSize) const {
    using cufft_type = typename CuFFTPrecisionAdapter<CalculatePrecision>::cufft_type;
    auto d_filter = make_cuda_sync_ptr<cufft_type>(__geometry.totalVoxelNum());

    process::initializeFouriorFilter<CalculatePrecision>(d_filter.get(), __geometry.voxelNum.x, __geometry.voxelNum.y);
    process::FouriorCutoffFilter<CalculatePrecision> ff{d_filter.get(), __geometry.voxelNum.x, __geometry.voxelNum.y,
                                                        __fouriorCutoffSize};
    for_each(
        __angleNum,
        [&](std::size_t i) {
          process::fouriorFilter(&__d_projectionPtrs_in[i * __geometry.totalVoxelNum()],
                                 &__d_projectionPtrs_out[i * __geometry.totalVoxelNum()], __geometry.voxelNum, ff);
        },
        cpu_threads.singleThread());
  }
  template <FloatingPoint_c CalculatePrecision = float>
  struct projection {
    CalculatePrecision const *projections_in;
    CBCTProjectionInfo const *projectionInfo;
    CalculatePrecision *image3d_out;
    basic::Image3DGeometry img_geometry;
    basic::Image2DGeometry proj_geometry;
    int angleNum;

    __host__ __device__ projection(
        CalculatePrecision const *__projections_in, CBCTProjectionInfo const *__projectionInfo,
        CalculatePrecision *__image3d_out, basic::Image3DGeometry __img_geometry,
        basic::Image2DGeometry __proj_geometry, int __angleNum)
        : projections_in(__projections_in)
        , projectionInfo(__projectionInfo)
        , image3d_out(__image3d_out)
        , img_geometry(__img_geometry)
        , proj_geometry(__proj_geometry)
        , angleNum(__angleNum) {}

    __device__ void operator()(
        std::size_t idx) const {
      // 手动计算 3D 坐标，而不是使用不存在的 at() 方法
      const auto X = static_cast<size_t>(idx % img_geometry.voxelNum.x);
      const auto Y = static_cast<size_t>((idx / img_geometry.voxelNum.x) % img_geometry.voxelNum.y);
      const auto Z = static_cast<size_t>(idx / (img_geometry.voxelNum.x * img_geometry.voxelNum.y));

      image3d_out[idx] += fdkProjectionNew_impl<CalculatePrecision>(
          projections_in, projectionInfo, angleNum, img_geometry, proj_geometry,
          math::InterpolationBilinear2D<CalculatePrecision>(), X, Y, Z);
    }
  };

private:
  template <typename T>
  std::vector<T *> pointer_by_steps(
      T *__begin, std::size_t __step, std::size_t __count) const {
    std::vector<T *> result;
    result.reserve(__count);
    for (std::size_t i = 0; i < __count; ++i)
      result.push_back(__begin + i * __step);
    return result;
  }

public:
  template <typename ProjectionValueType, FloatingPoint_c CalculatePrecision = float>
  void operator()(
      const CBCTDataView<ProjectionValueType> &__dataView, CalculatePrecision *__d_img_out,
      const basic::Image3DGeometry &__outGeometry) const {
    const auto projectionLocalGeometry =
        basic::make_ImageSizeByCenter(__dataView.pixelSize, basic::make_vec2<float>(0, 0), __dataView.pixels);
    const auto projectionTotalPixelNum = projectionLocalGeometry.totalVoxelNum() * __dataView.projectionNum;
    auto d_projections1 = make_cuda_sync_ptr<CalculatePrecision>(projectionTotalPixelNum);
    auto d_projections2 = make_cuda_sync_ptr<CalculatePrecision>(projectionTotalPixelNum);

    auto swap = [&] { std::swap(d_projections1, d_projections2); };
    // Do Air Correction
    {
      auto d_projectionDataPtrs = make_cuda_sync_ptr_from_hcopy(
          std::span<ProjectionValueType const *const>(__dataView.projectionDataPtrs, __dataView.projectionNum));
      auto d_airDataPtrs = make_cuda_sync_ptr_from_hcopy(
          std::span<ProjectionValueType const *const>(__dataView.airDataPtrs, __dataView.projectionNum));
      for_each_CUDA(projectionTotalPixelNum,
                    air(d_projectionDataPtrs.get(), d_airDataPtrs.get(), projectionLocalGeometry.totalVoxelNum(),
                        d_projections2.get()),
                    cudaStreamDefault);
      swap();
    }
    // Do Geometry Correction
    d_projections2.allocator().memset(0, d_projections2.span());
    for_each_CUDA(projectionTotalPixelNum,
                  geometry(d_projections1.get(), d_projections2.get(), projectionLocalGeometry, __dataView.geo_offsetU,
                           __dataView.geo_offsetV, __dataView.geo_angle,
                           math::InterpolationBilinear2D<CalculatePrecision>()),
                  cudaStreamDefault);
    swap();
    // Do Beam Hardening Correction
    for_each_CUDA(projectionTotalPixelNum,
                  BeamHardenMethods::v2(d_projections1.get(), d_projections2.get(), __dataView.beamHardenParamA,
                                        __dataView.beamHardenParamB),
                  cudaStreamDefault);
    swap();
    // Do Weight Correction
    for_each_CUDA(projectionTotalPixelNum,
                  weight(d_projections1.get(), d_projections2.get(), projectionLocalGeometry, __dataView.geo_SDD),
                  cudaStreamDefault);
    swap();
    // Do Fourier Filter
    fouriorFilterCUDA(d_projections1.data(), d_projections2.data(), projectionLocalGeometry, __dataView.projectionNum,
                      __dataView.fouriorCutoffLength);
    swap();
    // Do Projection
    for_each_CUDA(__outGeometry.totalVoxelNum(),
                  projection(d_projections1.get(), __dataView.projectionInfo, __d_img_out, __outGeometry,
                             projectionLocalGeometry, __dataView.projectionNum),
                  cudaStreamDefault);
  }
};
inline constexpr _FDK_CUDA FDK_CUDA{};
} // namespace
  // openpni::process
