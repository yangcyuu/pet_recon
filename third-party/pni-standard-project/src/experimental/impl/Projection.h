#pragma once

#include <cuda_runtime.h>
#include <span>

#include "include/experimental/algorithms/CalGeometry.hpp"
#include "include/experimental/algorithms/PathIntegral.hpp"
#include "include/experimental/core/Random.hpp"
#include "include/experimental/core/Span.hpp"
#include "include/experimental/node/LORBatch.hpp"
#include "include/experimental/node/MichCrystal.hpp"
#include "include/experimental/node/MichNorm.hpp"
#include "include/experimental/node/MichRandom.hpp"
#include "include/experimental/node/MichScatter.hpp"
#include "include/experimental/tools/Consts.hpp"
#include "include/experimental/tools/Parallel.hpp"
namespace openpni::experimental::node::impl {
__PNI_CUDA_MACRO__ inline float instant_path_integral(
    float __bias, core::TensorDataInput<float, 3> __img, core::Vector<float, 3> __cry_point1,
    core::Vector<float, 3> __cry_point2) {
  auto roi = core::cube_absolute(__img.grid.bounding_box());
  auto [amin, amax] = algorithms::liang_barskey_3d(roi, __cry_point1, __cry_point2);
  if (amax <= amin)
    return 0.0f;

  core::DirectionalLineSegment<float, 3> line_segment =
      core::DirectionalLineSegment<float, 3>::create_by_two_points_and_range(__cry_point1, __cry_point2, amin, amax);
  constexpr float sample_rate = 1.5f; // 每个体素采样点数
  const int sample_num =
      core::FMath<int>::max(3, line_segment.getLength() / __img.grid.spacing.l1() * tools::sqrt3 * sample_rate);
  core::SamplingUniform<float> sampler =
      core::SamplingUniform<float>::create(0, 1, sample_num, line_segment.getLength(), __bias);
  core::InterpolationNearestGetter<float, 3> interpolator(__img);
  return algorithms::calculate_path_integrals_impl(line_segment, __img, sampler, interpolator);
}
__PNI_CUDA_MACRO__ inline float senmap_fix_value(
    float __senmapValue, float __maxValue) {
  if (__senmapValue < __maxValue * 1e-4f)
    return __maxValue * 1e-3f;
  if (__senmapValue < __maxValue * 1e-2f)
    return __maxValue * 1e-2f * core::FMath<float>::fsqrt(__senmapValue / (__maxValue * 1e-2f));
  return __senmapValue;
}

__PNI_CUDA_MACRO__ inline float simple_path_integral(
    float bias, float sample_rate, core::Image3DInput<float> img, const core::Vector<float, 3> &point1,
    const core::Vector<float, 3> &point2, cubef __roi) {
  auto [amin, amax] = algorithms::liang_barskey_3d(__roi, point1, point2);
  if (amax <= amin)
    return 0;

  core::DirectionalLineSegment<float, 3> line_segment =
      core::DirectionalLineSegment<float, 3>::create_by_two_points_and_range(point1, point2, amin, amax);
  const int sample_num =
      core::FMath<int>::max(3, line_segment.getLength() / img.grid.spacing.l1() * tools::sqrt3 * sample_rate);
  core::SamplingUniform<float> sampler =
      core::SamplingUniform<float>::create(0, 1, sample_num, line_segment.getLength(), bias);
  core::InterpolationNearestGetter<float, 3> interpolator(img);
  return algorithms::calculate_path_integrals_impl(line_segment, img, sampler, interpolator);
}

__PNI_CUDA_MACRO__ inline float simple_path_integral(
    float bias, float sample_rate, core::Image3DInput<float> img, const core::Vector<float, 3> &point1,
    const core::Vector<float, 3> &point2) {
  return simple_path_integral(bias, sample_rate, img, point1, point2, core::cube_absolute(img.grid.bounding_box()));
}
__PNI_CUDA_MACRO__ inline void simple_reverse_path_integral(
    float bias, float sample_rate, float value, core::Image3DOutput<float> img, const core::Vector<float, 3> &point1,
    const core::Vector<float, 3> &point2, cubef __roi) {
  auto [amin, amax] = algorithms::liang_barskey_3d(__roi, point1, point2);
  if (amax <= amin)
    return;

  core::DirectionalLineSegment<float, 3> line_segment =
      core::DirectionalLineSegment<float, 3>::create_by_two_points_and_range(point1, point2, amin, amax);
  const int sample_num =
      core::FMath<int>::max(3, line_segment.getLength() / img.grid.spacing.l1() * tools::sqrt3 * sample_rate);
  core::SamplingUniform<float> sampler =
      core::SamplingUniform<float>::create(0, 1, sample_num, line_segment.getLength(), bias);
  core::InterpolationNearestSetter<float, 3> interpolator(img);
  algorithms::calculate_path_reverse_integrals_impl(line_segment, img, value, sampler, interpolator);
}
__PNI_CUDA_MACRO__ inline void simple_reverse_path_integral(
    float bias, float sample_rate, float value, core::Image3DOutput<float> img, const core::Vector<float, 3> &point1,
    const core::Vector<float, 3> &point2) {
  simple_reverse_path_integral(bias, sample_rate, value, img, point1, point2,
                               core::cube_absolute(img.grid.bounding_box()));
}
__PNI_CUDA_MACRO__ inline void simple_reverse_path_integral(
    float bias, float sample_rate, core::Image3DOutput<float> img, const core::MichStandardEvent &event) {
  simple_reverse_path_integral(bias, sample_rate, event.value, img, event.geo1.O, event.geo2.O);
}

__PNI_CUDA_MACRO__ inline float simple_path_integral_TOF(
    float bias, float sample_rate, core::Image3DInput<float> img, const core::Vector<float, 3> &point1,
    const core::Vector<float, 3> &point2, core::Vector<int16_t, 2> &pair_tofMean, core::Vector<int16_t, 2> &pair_tofDev,
    int16_t deltaTOF, int16_t TOFBinWid_ps, cubef __roi) {
  auto [amin, amax] = algorithms::liang_barskey_3d(__roi, point1, point2);
  if (amax <= amin)
    return 0;

  core::DirectionalLineSegment<float, 3> line_segment =
      core::DirectionalLineSegment<float, 3>::create_by_two_points(point1, point2);
  const int sample_num =
      core::FMath<int>::max(3, line_segment.getLength() / img.grid.spacing.l1() * tools::sqrt3 * sample_rate);
  float distance_cry1_cry2 = algorithms::l2(point2 - point1);
  core::SamplingUniformWithTOF<float> sampler = core::SamplingUniformWithTOF<float>::create(
      amin, amax, sample_num, distance_cry1_cry2, pair_tofMean, pair_tofDev, deltaTOF, TOFBinWid_ps, bias);
  if (amin > sampler.get_alpha_max() || amax < sampler.get_alpha_min())
    return 0;
  core::InterpolationNearestGetter<float, 3> interpolator(img);
  return algorithms::calculate_path_integrals_impl(line_segment, img, sampler, interpolator);
}

__PNI_CUDA_MACRO__ inline void simple_reverse_path_integral_TOF(
    float bias, float sample_rate, float value, core::Image3DOutput<float> img, const core::Vector<float, 3> &point1,
    const core::Vector<float, 3> &point2, core::Vector<int16_t, 2> &pair_tofMean, core::Vector<int16_t, 2> &pair_tofDev,
    int16_t deltaTOF, uint16_t TOFBinWid_ps, cubef __roi) {
  auto [amin, amax] = algorithms::liang_barskey_3d(__roi, point1, point2);
  if (amax <= amin)
    return;

  core::DirectionalLineSegment<float, 3> line_segment =
      core::DirectionalLineSegment<float, 3>::create_by_two_points(point1, point2);
  const int sample_num =
      core::FMath<int>::max(3, line_segment.getLength() / img.grid.spacing.l1() * tools::sqrt3 * sample_rate);
  float distance_cry1_cry2 = algorithms::l2(point2 - point1);
  core::SamplingUniformWithTOF<float> sampler = core::SamplingUniformWithTOF<float>::create(
      amin, amax, sample_num, distance_cry1_cry2, pair_tofMean, pair_tofDev, deltaTOF, TOFBinWid_ps, bias);
  core::InterpolationNearestSetter<float, 3> interpolator(img);
  algorithms::calculate_path_reverse_integrals_impl(line_segment, img, value, sampler, interpolator);
}

} // namespace openpni::experimental::node::impl
namespace openpni::experimental::node::impl {
void h_gen_some_attn_factors(float *__out_factors, std::span<core::MichStandardEvent const> __events,
                             float const *__attn_map, core::Grids<3> __map);
void h_gen_some_attn_factors(float *__out_factors, std::span<core::MichStandardEvent const> __events,
                             const float *__attn_map, core::Grids<3> __map);
float h_cal_update_measurements(float const *h_updateImage, float const *h_senmapImage, std::size_t count);
void h_apply_correction_factor(std::span<float> values, std::span<core::MichStandardEvent> _event,
                               node::MichNormalization *michNorm, node::MichRandom *michRand,
                               node::MichScatter *michScat);
} // namespace openpni::experimental::node::impl
namespace openpni::experimental::node::impl {
void d_vector_divide(std::span<float> __data, float __divisor);
void d_image_update(float const *d_updateImage, float *d_out, float const *d_senmap, int64_t size);
void d_fill_standard_events_ids_from_lor_ids(core::MichStandardEvent *__events, std::span<std::size_t const> __lorIds,
                                             core::MichDefine __michDefine);
void d_fill_standard_events_ids_from_listmode(core::MichStandardEvent *__events,
                                              std::span<openpni::basic::Listmode_t const> __listmode_data,
                                              core::MichDefine __michDefine);
void d_fill_standard_events_ids_from_listmodeTOF(core::MichStandardEvent *__events,
                                                 std::span<openpni::basic::Listmode_t const> __listmode_data,
                                                 int16_t TOF_division, core::MichDefine __michDefine);
void d_fill_standard_events_values(core::MichStandardEvent *__events, float const *__values, std::size_t __count);
void d_mul_standard_events_values(core::MichStandardEvent *__events, float const *__values, std::size_t __count);
void d_fix_senmap_value(std::span<float> __d_data);
void d_gen_some_attn_factors(float *__out_factors, std::span<core::MichStandardEvent const> __events,
                             float const *__attn_map, core::Grids<3> __map);
void d_redirect_to_mich(std::size_t __count, std::size_t const *__d_lorId, float const *__d_factors,
                        float *__d_out_factors);
void d_redirect_from_mich(std::size_t __count, std::size_t const *__d_lorId, float const *__d_factors,
                          float *__d_out_factors);
void d_fill_factors_from_mich(std::size_t __count, core::MichStandardEvent const *__d_events, float *__out_factors,
                              float const *__michValues, core::MichDefine __mich);
void d_apply_correction_factor(float *d_values, std::span<core::MichStandardEvent> d_events,
                               node::MichNormalization *michNorm, node::MichRandom *michRand,
                               node::MichScatter *michScat, node::MichAttn *michAttn, cuda_sync_ptr<float> &d_addFactor,
                               cuda_sync_ptr<float> &d_mulFactor);
void d_osem_fix_integral_value(std::span<float> __d_data);
void d_simple_path_integral_batch(core::TensorDataInput<float, 3> __img,
                                  std::span<core::MichStandardEvent const> __events, float __sample_rate,
                                  float *__out_values);
void d_simple_path_reverse_integral_batch(core::TensorDataOutput<float, 3> __img,
                                          std::span<core::MichStandardEvent const> __events, float __sample_rate);
void d_simple_path_integral_batch_TOF(core::TensorDataInput<float, 3> __img,
                                      std::span<core::MichStandardEvent const> __events, float __sample_rate,
                                      int16_t TOFBinWid_ps, float *__out_values);
void d_simple_path_reverse_integral_batch_TOF(core::TensorDataOutput<float, 3> __img,
                                              std::span<core::MichStandardEvent const> __events, float __sample_rate,
                                              int16_t TOFBinWid_ps);
float d_cal_update_measurements(float const *__updateImage, float const *__senmapImage, std::size_t count);
void d_count_from_listmode(std::span<basic::Listmode_t const> __listmodes, std::size_t const *__lorids,
                           core::MichDefine __michDefine, float *__out_counts);
void d_redirect_from_mich_from_slice_range(std::size_t __sliceBegin, std::size_t __sliceEnd,
                                           float const *__d_michValues, float *__d_outValues,
                                           core::MichDefine __michDefine);
void d_redirect_from_mich_from_slice_range(std::size_t __sliceBegin, std::size_t __sliceEnd,
                                           std::span<basic::Listmode_t const> __d_listmodes, float *__d_outValues,
                                           core::MichDefine __michDefine);
} // namespace openpni::experimental::node::impl