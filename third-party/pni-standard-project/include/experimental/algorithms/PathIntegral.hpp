#pragma once
#include "../core/Geometric.hpp"
#include "../core/Interpolation.hpp"
#include "../core/Sampling.hpp"
namespace openpni::experimental::algorithms {

template <FloatingPoint_c F, int N, typename SampMethod, typename InterpolationGetter>
__PNI_CUDA_MACRO__ inline F calculate_path_integrals_impl(
    core::DirectionalLineSegment<F, N> __line_segment, openpni::experimental::core::TensorDataInput<F, N> __Img,
    SampMethod __sampler, InterpolationGetter __interpolator) {
  F integral_sum{0};
  while (__sampler.has_next()) {
    const auto [t, step] = __sampler.next();                  // 获取下一个采样点参数t
    const auto sample_point_vec = __line_segment.getPoint(t); // 使用有向线段直接获取采样点
    F value = __interpolator.get(sample_point_vec);           // 在采样点处进行插值，获取图像值
    integral_sum += value * step;                             // 累加到积分和中（值 * 步长）
  }
  return integral_sum;
}

template <typename LineSegment, typename TensorInput, typename PathValue, typename SampMethod,
          typename InterpolationSetter>
__PNI_CUDA_MACRO__ inline void calculate_path_reverse_integrals_impl(
    LineSegment __line_segment, TensorInput __Img, PathValue __pathValue, SampMethod __sampler,
    InterpolationSetter __interpolator) {
  while (__sampler.has_next()) {
    const auto [t, step] = __sampler.next();                  // 获取下一个采样点参数t
    const auto sample_point_vec = __line_segment.getPoint(t); // 使用有向线段直接获取采样点
    __interpolator.add(sample_point_vec, __pathValue * step); // 反向投影（值 * 步长）
  }
}

template <typename SampMethod = core::SamplingUniform<float>,
          typename InterpolationGetter = core::InterpolationNearestGetter<float, 3>>
__PNI_CUDA_MACRO__ inline float calculatePathIntegrals_impl(
    size_t i, const Vec3<float> &point1, const Vec3<float> &point2,
    const openpni::experimental::core::TensorDataInput<float, 3> &Img, const cube<float> &roi, const int sampleNum) {
  const auto a = algorithms::liang_barskey_3d(roi, point1, point2);
  const auto amin = a[0];
  const auto amax = a[1];
  if (amax <= amin) {
    return float(0);
  }

  auto line_segment = core::DirectionalLineSegment<float, 3>::create_by_two_points(point1, point2);

  auto clipped_start = line_segment.getPoint(amin);
  auto clipped_end = line_segment.getPoint(amax);

  auto clipped_segment = core::DirectionalLineSegment<float, 3>::create_by_two_points(clipped_start, clipped_end);
  const auto segment_length = clipped_segment.getLength();
  const auto stepSize = segment_length / sampleNum;

  // 初始化采样器和插值器
  float sampleBias = static_cast<float>((i * 2654435761U) % 10000) / 10000.0f; // 取0～1的伪随机数
  auto sampler = SampMethod::create(0.0f, 1.0f, sampleNum, sampleBias);
  InterpolationGetter interpolator_getter(Img);

  float integral_sum = 0;
  while (sampler.has_next()) {
    // 获取下一个采样点参数t
    const auto sample_result = sampler.next();
    const auto t = sample_result[0];

    // 使用有向线段直接获取采样点
    const auto sample_point_vec = clipped_segment.getPoint(t);

    // 在采样点处进行插值，获取图像值
    float value = interpolator_getter.get(sample_point_vec);

    // 累加到积分和中（值 * 步长）
    integral_sum += value * stepSize;
  }
  return integral_sum;
}

template <typename SampMethod = core::SamplingUniform<float>,
          typename InterpolationMethod = core::InterpolationNearest<float, 3>>
__PNI_CUDA_MACRO__ inline void calculateAntiPathIntegrals_impl(
    size_t i, const Vec3<float> &point1, const Vec3<float> &point2, float pathValue,
    openpni::experimental::core::TensorDataIO<float, 3> &Img, const cube<float> &roi, const int sampleNum) {
  const auto a = algorithms::liang_barskey_3d(roi, point1, point2);
  const auto amin = a[0];
  const auto amax = a[1];
  if (amax <= amin) {
    return;
  }

  auto line_segment = core::DirectionalLineSegment<float, 3>::create_by_two_points(point1, point2);

  auto clipped_start = line_segment.getPoint(amin);
  auto clipped_end = line_segment.getPoint(amax);

  auto clipped_segment = core::DirectionalLineSegment<float, 3>::create_by_two_points(clipped_start, clipped_end);
  const auto segment_length = clipped_segment.getLength();
  const auto stepSize = segment_length / sampleNum;

  // 初始化采样器和插值器
  float sampleBias = static_cast<float>((i * 2654435761U) % 10000) / 10000.0f; // 取0～1的伪随机数
  auto sampler = SampMethod::create(0.0f, 1.0f, sampleNum, sampleBias);
  InterpolationMethod interpolator(Img);

  while (sampler.has_next()) {
    // 获取下一个采样点参数t
    const auto sample_result = sampler.next();
    const auto t = sample_result[0];

    // 使用有向线段直接获取采样点
    const auto sample_point_vec = clipped_segment.getPoint(t);

    // 累加到积分和中（值 * 步长）
    interpolator.add(sample_point_vec, pathValue * stepSize);
  }
}
} // namespace openpni::experimental::algorithms