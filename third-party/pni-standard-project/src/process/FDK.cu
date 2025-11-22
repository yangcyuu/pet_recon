#include "../../include/basic/Math.hpp"
#include "../../include/math/FouriorFilter.cuh"
#include "../../include/math/Transforms.cuh"
#include "../../include/process/FDK.hpp"
#include "../../include/process/Foreach.cuh"
namespace openpni::process {

// air struct implementation
__host__ __device__ _FDK_CUDA::air::air(
    float const *const *__projectionPtrs, float const *const *__airValuePtrs, std::size_t __crystalNum,
    float *__outProjectionValue)
    : projectionPtrs(__projectionPtrs)
    , airValuePtrs(__airValuePtrs)
    , crystalNum(__crystalNum)
    , outProjectionValue(__outProjectionValue) {}

__host__ __device__ void _FDK_CUDA::air::operator()(
    std::size_t idx) const {
  using fmath = basic::FMath<float>;
  const auto projectionId = idx / crystalNum;
  const auto pixelId = idx % crystalNum;
  outProjectionValue[idx] = float(-1.) * fmath::flog(float(projectionPtrs[projectionId][pixelId]) /
                                                     float(airValuePtrs[projectionId][pixelId]));
}

// geometry struct implementation
__host__ __device__ _FDK_CUDA::geometry::geometry(
    float const *__projections_in, float *__projections_out, basic::Image2DGeometry __geometry, float __offsetU,
    float __offsetV, float __angle, const math::InterpolationBilinear2D<float> &__ipMethod)
    : projections_in(__projections_in)
    , projections_out(__projections_out)
    , geometry2D(__geometry)
    , offsetU(__offsetU)
    , offsetV(__offsetV)
    , angle(__angle)
    , ipMethod(__ipMethod)
    , actionTranslateProjection{basic::make_vec2<float>(-__offsetU, -__offsetV)}
    , actionRotateProjection{basic::make_vec2<float>((__geometry.voxelNum - 1) / 2.0),
                             basic::rotationMatrix<float>(__angle)} {}

__device__ void _FDK_CUDA::geometry::operator()(
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

// BeamHardenMethods::v1 implementation
__host__ __device__ _FDK_CUDA::BeamHardenMethods::v1::v1(
    float const *__inProjections, float *__outProjections)
    : inProjections(__inProjections)
    , outProjections(__outProjections) {}

__host__ __device__ void _FDK_CUDA::BeamHardenMethods::v1::operator()(
    std::size_t idx) const {
  using fmath = basic::FMath<float>;
  outProjections[idx] = inProjections[idx] < float(1e-6)
                            ? float(0)
                            : fmath::fpow(fmath::abs(fmath::fexp(inProjections[idx]) - float(1)), float(0.8));
}

// BeamHardenMethods::v2 implementation
__host__ __device__ _FDK_CUDA::BeamHardenMethods::v2::v2(
    float const *__inProjections, float *__outProjections, float __paramA, float __paramB)
    : inProjections(__inProjections)
    , outProjections(__outProjections)
    , paramA(__paramA)
    , paramB(__paramB) {}

__host__ __device__ void _FDK_CUDA::BeamHardenMethods::v2::operator()(
    std::size_t idx) const {
  using fmath = basic::FMath<float>;
  const auto &temp = inProjections[idx];
  outProjections[idx] = temp + paramA * temp * temp + paramB * temp * temp * temp;
}

// weight struct implementation
__host__ __device__ _FDK_CUDA::weight::weight(
    float const *__projections_in, float *__projections_out, basic::Image2DGeometry __geometry, float __SDD)
    : projections_in(__projections_in)
    , projections_out(__projections_out)
    , geometry(__geometry)
    , SDD(__SDD) {}

__host__ __device__ void _FDK_CUDA::weight::operator()(
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

// fouriorFilterCUDA implementation
void _FDK_CUDA::fouriorFilterCUDA(
    float const *__d_projectionPtrs_in, // p = host, *p = host, **p = device
    float *__d_projectionPtrs_out,      // p = host, *p = host, **p = device
    basic::Image2DGeometry __geometry, int __angleNum, unsigned __fouriorCutoffSize) const {
  using cufft_type = typename CuFFTPrecisionAdapter<float>::cufft_type;
  auto d_filter = make_cuda_sync_ptr<cufft_type>(__geometry.totalVoxelNum());

  process::initializeFouriorFilter<float>(d_filter.get(), __geometry.voxelNum.x);
  process::FouriorCutoffFilter<float> ff{d_filter.get(), __geometry.voxelNum.x, __geometry.voxelNum.y,
                                         __fouriorCutoffSize};
  for_each(
      __angleNum,
      [&](std::size_t i) {
        process::fouriorFilter(&__d_projectionPtrs_in[i * __geometry.totalVoxelNum()],
                               &__d_projectionPtrs_out[i * __geometry.totalVoxelNum()], __geometry.voxelNum, ff);
      },
      cpu_threads.singleThread());
}

// projection struct implementation
__host__ __device__ _FDK_CUDA::projection::projection(
    float const *__projections_in, CBCTProjectionInfo const *__projectionInfo, float *__image3d_out,
    basic::Image3DGeometry __img_geometry, basic::Image2DGeometry __proj_geometry, int __angleNum)
    : projections_in(__projections_in)
    , projectionInfo(__projectionInfo)
    , image3d_out(__image3d_out)
    , img_geometry(__img_geometry)
    , proj_geometry(__proj_geometry)
    , angleNum(__angleNum) {}

__host__ __device__ void _FDK_CUDA::projection::operator()(
    std::size_t idx) const {
  // 手动计算 3D 坐标，而不是使用不存在的 at() 方法
  const auto X = static_cast<size_t>(idx % img_geometry.voxelNum.x);
  const auto Y = static_cast<size_t>((idx / img_geometry.voxelNum.x) % img_geometry.voxelNum.y);
  const auto Z = static_cast<size_t>(idx / (img_geometry.voxelNum.x * img_geometry.voxelNum.y));

  image3d_out[idx] += fdkProjectionNew_impl(projections_in, projectionInfo, angleNum, img_geometry, proj_geometry,
                                            math::InterpolationBilinear2D<float>(), X, Y, Z);
}

// _FDK_CUDA::operator() implementation
void _FDK_CUDA::operator()(
    const CBCTDataView &__dataView, float *__d_img_out, const basic::Image3DGeometry &__outGeometry) const {
  const auto projectionLocalGeometry =
      basic::make_ImageSizeByCenter(__dataView.pixelSize, basic::make_vec2<float>(0, 0), __dataView.pixels);
  const auto projectionTotalPixelNum = projectionLocalGeometry.totalVoxelNum() * __dataView.projectionNum;
  auto d_projections1 = make_cuda_sync_ptr<float>(projectionTotalPixelNum);
  auto d_projections2 = make_cuda_sync_ptr<float>(projectionTotalPixelNum);

  auto swap = [&] { std::swap(d_projections1, d_projections2); };

  // Do Air Correction
  {
    auto d_projectionDataPtrs = make_cuda_sync_ptr_from_hcopy(
        std::span<float const *const>(__dataView.projectionDataPtrs, __dataView.projectionNum));
    auto d_airDataPtrs =
        make_cuda_sync_ptr_from_hcopy(std::span<float const *const>(__dataView.airDataPtrs, __dataView.projectionNum));
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
                         __dataView.geo_offsetV, __dataView.geo_angle, math::InterpolationBilinear2D<float>()),
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

// FDK 后处理函数实现 - 归一化、CT值缩放、CO校正和Z轴翻转
void FDK_PostProcessing(
    cuda_sync_ptr<float> &d_imgOut, const basic::Image3DGeometry &outputGeometry, const io::U16Image &ctImage,
    float ct_slope, float ct_intercept, float co_offset_x, float co_offset_y) {
  // 创建无符号整数版本的体素数
  auto voxelNumUnsigned = basic::make_vec3<unsigned>(static_cast<unsigned>(outputGeometry.voxelNum.x),
                                                     static_cast<unsigned>(outputGeometry.voxelNum.y),
                                                     static_cast<unsigned>(outputGeometry.voxelNum.z));

  // 计算归一化因子（投影数量的倒数乘以360度）
  float normalizeScale = 360.0f / ctImage.imageGeometry().voxelNum.z;

  // 步骤1：归一化处理
  for_each_CUDA(outputGeometry.totalVoxelNum(), CBCTNormalize<float, float>{normalizeScale, d_imgOut.get()},
                cudaStreamDefault);

  // 步骤2：CT值后处理（应用ct_slope和ct_intercept）
  for_each_CUDA(outputGeometry.totalVoxelNum(),
                CBCTPostProcess<float, float>{ct_slope, ct_intercept, voxelNumUnsigned, d_imgOut.get()},
                cudaStreamDefault);

  // 步骤3：中心偏移校正（CO correction）
  for_each_CUDA(outputGeometry.totalVoxelNum(),
                CBCTCO<float, float>{co_offset_x, co_offset_y, voxelNumUnsigned, outputGeometry.voxelSize,
                                     basic::make_vec3<float>(0.0f, 0.0f, 0.0f), d_imgOut.get()},
                cudaStreamDefault);

  // 步骤4：Z轴翻转处理
  io::F32Image outputImage(outputGeometry);
  auto tempHostData = std::make_unique<float[]>(outputGeometry.totalVoxelNum());
  d_imgOut.allocator().copy_from_device_to_host(tempHostData.get(), d_imgOut.cspan());

  CBCTFlipZ<float> flipZ{tempHostData.get(), static_cast<unsigned>(outputGeometry.voxelNum.x),
                         static_cast<unsigned>(outputGeometry.voxelNum.y),
                         static_cast<unsigned>(outputGeometry.voxelNum.z)};
  for (size_t i = 0; i < outputGeometry.totalVoxelNum(); ++i) {
    outputImage.data()[i] = flipZ(i);
  }

  // 将处理结果复制回设备内存
  d_imgOut.allocator().copy_from_host_to_device(d_imgOut.get(), outputImage.cspan());
}

} // namespace openpni::process
