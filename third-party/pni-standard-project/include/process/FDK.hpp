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

#include "../basic/CudaPtr.hpp"
#include "../basic/Image.hpp"
#include "../basic/Math.hpp"
#include "../io/IO.hpp"
#include "../math/Convolution.hpp"
#include "../math/Geometry.hpp"
#include "../math/Interpolation.hpp"
#include "../math/Transforms.hpp"
#include "Foreach.hpp"
namespace openpni::process {
struct CBCTProjectionInfo {
  basic::Vec3<float> positionX;  // X射线源的3D位置
  basic::Vec3<float> directionU; // 探测器平面U方向的单位向量
  basic::Vec3<float> directionV; // 探测器平面V方向的单位向量
  basic::Vec3<float> positionD;  // 探测器平面上中心点的3D位置
};
struct CBCTDataView {
  CBCTProjectionInfo *projectionInfo;
  float const *const *projectionDataPtrs; // 投影图值，一级指针在内存，数据在内存或GPU上
  float const *const *airDataPtrs;        // 空气扫描值，一级指针在内存，数据在内存或GPU上
  int projectionNum;                      // 有多少张投影图
  basic::Vec2<unsigned> pixels;           // 投影图像素数量
  basic::Vec2<float> pixelSize;           // 投影图每个像素的大小
  float geo_angle;                        // 几何校正里面的角度
  float geo_offsetU;                      // 几何校正里面的U方向偏移
  float geo_offsetV;                      // 几何校正里面的V方向偏移
  float geo_SDD;                          // 几何校正里面的源到探测器距离
  float geo_SOD;                          // 几何校正里面的源到中心点距离
  unsigned fouriorCutoffLength;           // 傅立叶平滑核的截止长度
  float beamHardenParamA;                 // 束硬化校正参数A
  float beamHardenParamB;                 // 束硬化校正参数B
};

template <typename InterpolationMethod>
inline __PNI_CUDA_MACRO__ float fdkProjectionNew_impl(
    float const *__projections, const CBCTProjectionInfo *__projectionInfo, int __projectionNum,
    basic::Image3DGeometry __img3dGeometry, basic::Image2DGeometry __img2dGeometry, InterpolationMethod __ipMethod,
    unsigned __X, unsigned __Y, unsigned __Z) {
  float sum = 0;
  const auto voxelPosition = __img3dGeometry.voxel_center(basic::make_vec3<unsigned>(__X, __Y, __Z));
  for (int i = 0; i < __projectionNum; i++) {
    const auto &XPosition = __projectionInfo[i].positionX;               // X射线源的3D位置
    const auto UDirection = __projectionInfo[i].directionU.normalized(); // 探测器平面U方向的单位向量
    const auto VDirection = __projectionInfo[i].directionV.normalized(); // 探测器平面V方向的单位向量
    const auto &DPosition = __projectionInfo[i].positionD;               // 探测器平面上中心点的3D位置
    const auto line_from_X_to_voxel = basic::Line<float>::create_from_ends(XPosition, voxelPosition);
    const auto plane_of_detector = basic::Plane<float>::create(DPosition, UDirection, VDirection);
    const auto intersection = basic::intersection(line_from_X_to_voxel, plane_of_detector);
    const auto intersectionInProjection_wrt_detectorCenter =
        basic::make_vec2<float>(intersection.u_plane,
                                intersection.v_plane); // 交点相对于探测器中心的坐标
    sum += __ipMethod(intersectionInProjection_wrt_detectorCenter, &__projections[i * __img2dGeometry.totalVoxelNum()],
                      __img2dGeometry);
  }
  return sum;
}

struct _FDK_CUDA {
  struct air {
    float const *const *projectionPtrs;
    float const *const *airValuePtrs;
    std::size_t crystalNum;
    float *outProjectionValue;
    __PNI_CUDA_MACRO__ air(float const *const *__projectionPtrs, float const *const *__airValuePtrs,
                           std::size_t __crystalNum, float *__outProjectionValue);
    __PNI_CUDA_MACRO__ void operator()(std::size_t idx) const;
  };

  struct geometry {
    float const *projections_in;
    float *projections_out;
    basic::Image2DGeometry geometry2D;
    float offsetU;
    float offsetV;
    float angle;
    math::InterpolationBilinear2D<float> ipMethod;

    process::ActionTranslate2D actionTranslateProjection;
    process::ActionRotate2D actionRotateProjection;

    __PNI_CUDA_MACRO__ geometry(float const *__projections_in, float *__projections_out,
                                basic::Image2DGeometry __geometry, float __offsetU, float __offsetV, float __angle,
                                const math::InterpolationBilinear2D<float> &__ipMethod);

    __PNI_CUDA_MACRO__ void operator()(std::size_t idx) const;
  };

  struct BeamHardenMethods {
    struct v1 {
      float const *inProjections;
      float *outProjections;

      __PNI_CUDA_MACRO__ v1(float const *__inProjections, float *__outProjections);

      __PNI_CUDA_MACRO__ void operator()(std::size_t idx) const;
    };
    struct v2 {
      float paramA;
      float paramB;
      float const *inProjections;
      float *outProjections;

      __PNI_CUDA_MACRO__ v2(float const *__inProjections, float *__outProjections, float __paramA, float __paramB);

      __PNI_CUDA_MACRO__ void operator()(std::size_t idx) const;
    };
  };

  struct weight {
    float const *projections_in;
    float *projections_out;
    basic::Image2DGeometry geometry;
    float SDD;

    __PNI_CUDA_MACRO__ weight(float const *__projections_in, float *__projections_out,
                              basic::Image2DGeometry __geometry, float __SDD);

    __PNI_CUDA_MACRO__ void operator()(std::size_t idx) const;
  };

  void fouriorFilterCUDA(float const *__d_projectionPtrs_in, // p = host, *p = host, **p = device
                         float *__d_projectionPtrs_out,      // p = host, *p = host, **p = device
                         basic::Image2DGeometry __geometry, int __angleNum, unsigned __fouriorCutoffSize) const;

  struct projection {
    float const *projections_in;
    CBCTProjectionInfo const *projectionInfo;
    float *image3d_out;
    basic::Image3DGeometry img_geometry;
    basic::Image2DGeometry proj_geometry;
    int angleNum;

    __PNI_CUDA_MACRO__ projection(float const *__projections_in, CBCTProjectionInfo const *__projectionInfo,
                                  float *__image3d_out, basic::Image3DGeometry __img_geometry,
                                  basic::Image2DGeometry __proj_geometry, int __angleNum);

    __PNI_CUDA_MACRO__ void operator()(std::size_t idx) const;
  };

public:
  void operator()(const CBCTDataView &__dataView, float *__d_img_out,
                  const basic::Image3DGeometry &__outGeometry) const;
};
inline constexpr _FDK_CUDA FDK_CUDA{};
void FDK_PostProcessing(cuda_sync_ptr<float> &d_imgOut, const basic::Image3DGeometry &outputGeometry,
                        const io::U16Image &ctImage, float ct_slope, float ct_intercept, float co_offset_x,
                        float co_offset_y);

} // namespace
  // openpni::process
