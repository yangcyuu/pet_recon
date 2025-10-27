#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>

#include "Projection.h"
#include "Test.h"
#include "include/experimental/example/EasyParallel.hpp"
#include "include/experimental/tools/Parallel.cuh"
namespace openpni::experimental::node::impl {
void d_vector_divide(
    std::span<float> __data, float __divisor) {
  tools::parallel_for_each_CUDA(__data.size(),
                                [=, __data = __data.data()] __device__(std::size_t idx) { __data[idx] /= __divisor; });
}
void d_image_update(
    float const *d_updateImage, float *d_out, float const *d_senmap, int64_t size) {
  tools::parallel_for_each_CUDA(size,
                                [=] __device__(std::size_t idx) { d_out[idx] *= d_updateImage[idx] / d_senmap[idx]; });
  auto max_value =
      thrust::reduce(thrust::cuda::par.on(basic::cuda_ptr::default_stream()), thrust::device_pointer_cast(d_out),
                     thrust::device_pointer_cast(d_out + size), 0.0f, thrust::maximum<float>());
  const auto min_value = max_value * 1e-7f;
  tools::parallel_for_each_CUDA(
      size, [=] __device__(std::size_t idx) { d_out[idx] = core::FMath<float>::max(d_out[idx], min_value); });
}
void d_fill_standard_events_ids_from_lor_ids(
    core::MichStandardEvent *__events, std::span<std::size_t const> __lorIds, core::MichDefine __michDefine) {
  const auto indexConverter = core::IndexConverter::create(__michDefine);
  tools::parallel_for_each_CUDA(__lorIds.size(), [=, __lorIds = __lorIds.data()] __device__(std::size_t index) {
    auto [cry1Rid, cry2Rid] = indexConverter.getCrystalIDFromLORID(__lorIds[index]);
    __events[index].crystal1 = cry1Rid;
    __events[index].crystal2 = cry2Rid;
    __events[index].tof = 0;
    __events[index].value = 1.0f;
  });
}
void d_fill_standard_events_ids_from_listmode(
    core::MichStandardEvent *__events, std::span<openpni::basic::Listmode_t const> __listmode_data,
    core::MichDefine __michDefine) {
  tools::parallel_for_each_CUDA(__listmode_data.size(), [=, listmode_in = __listmode_data.data(),
                                                         poly = __michDefine.polygon,
                                                         det = __michDefine.detector] __device__(std::size_t index) {
    const auto &listmode = listmode_in[index];
    auto uniformID1 = core::mich::getUniformIdFromFlatId(poly, det, static_cast<int>(listmode.globalCrystalIndex1));
    auto uniformID2 = core::mich::getUniformIdFromFlatId(poly, det, static_cast<int>(listmode.globalCrystalIndex2));
    __events[index].crystal1 = core::mich::getRectangleIDFromUniformID(poly, det, uniformID1);
    __events[index].crystal2 = core::mich::getRectangleIDFromUniformID(poly, det, uniformID2);

    __events[index].tof = listmode.time1_2pico;
    __events[index].value = 1.0f; // Listmode events have implicit value = 1
  });
}
void d_fill_standard_events_ids_from_listmodeTOF(
    core::MichStandardEvent *__events, std::span<openpni::basic::Listmode_t const> __listmode_data,
    int16_t TOF_division, core::MichDefine __michDefine) {
  tools::parallel_for_each_CUDA(__listmode_data.size(), [=, listmode_in = __listmode_data.data(),
                                                         poly = __michDefine.polygon,
                                                         det = __michDefine.detector] __device__(std::size_t index) {
    const auto &listmode = listmode_in[index];
    auto uniformID1 = core::mich::getUniformIdFromFlatId(poly, det, static_cast<int>(listmode.globalCrystalIndex1));
    auto uniformID2 = core::mich::getUniformIdFromFlatId(poly, det, static_cast<int>(listmode.globalCrystalIndex2));
    __events[index].crystal1 = core::mich::getRectangleIDFromUniformID(poly, det, uniformID1);
    __events[index].crystal2 = core::mich::getRectangleIDFromUniformID(poly, det, uniformID2);

    __events[index].tof = listmode.time1_2pico;
    __events[index].cry1_tof_deviation = TOF_division;
    __events[index].cry2_tof_deviation = TOF_division;
    __events[index].value = 1.0f; // Listmode events have implicit value = 1
  });
}
void d_fill_standard_events_values(
    core::MichStandardEvent *__events, float const *__values, std::size_t __count) {
  tools::parallel_for_each_CUDA(__count,
                                [=] __device__(std::size_t index) { __events[index].value = __values[index]; });
}
void d_mul_standard_events_values(
    core::MichStandardEvent *__events, float const *__values, std::size_t __count) {
  tools::parallel_for_each_CUDA(__count,
                                [=] __device__(std::size_t index) { __events[index].value *= __values[index]; });
}
void d_fix_senmap_value(
    std::span<float> __d_data) {
  if (__d_data.size() == 0)
    return;
  const auto maxValue = thrust::reduce(
      thrust::cuda::par.on(basic::cuda_ptr::default_stream()), thrust::device_pointer_cast(__d_data.data()),
      thrust::device_pointer_cast(__d_data.data() + __d_data.size()), 0.0f, thrust::maximum<float>());
  if (maxValue <= 0.0f)
    return;
  tools::parallel_for_each_CUDA(__d_data.size(), [=, __d_data = __d_data.data()] __device__(std::size_t idx) {
    __d_data[idx] = senmap_fix_value(__d_data[idx], maxValue);
  });
}
void d_gen_some_attn_factors(
    float *__out_factors, std::span<core::MichStandardEvent const> __events, float const *__attn_map,
    core::Grids<3> __map) {
  tools::parallel_for_each_CUDA(__events.size(), [=, __events = __events.data()] __device__(std::size_t index) {
    __out_factors[index] =
        instant_path_integral(core::instant_random_float(index), core::TensorDataInput<float, 3>{__map, __attn_map},
                              __events[index].geo1.O, __events[index].geo2.O);
    __out_factors[index] = core::FMath<float>::fexp(-__out_factors[index]);
  });
}

void d_redirect_to_mich(
    std::size_t __count, std::size_t const *__d_lorId, float const *__d_factors, float *__d_out_factors) {
  tools::parallel_for_each_CUDA(
      __count, [=] __device__(std::size_t index) { __d_out_factors[__d_lorId[index]] = __d_factors[index]; });
}
void d_redirect_from_mich(
    std::size_t __count, std::size_t const *__d_lorId, float const *__d_factors, float *__d_out_factors) {
  tools::parallel_for_each_CUDA(
      __count, [=] __device__(std::size_t index) { __d_out_factors[index] = __d_factors[__d_lorId[index]]; });
}
void d_fill_factors_from_mich(
    std::size_t __count, core::MichStandardEvent const *__d_events, float *__out_factors, float const *__michValues,
    core::MichDefine __mich) {
  auto indexConverter = core::IndexConverter::create(__mich);
  auto michSize = core::MichInfoHub::create(__mich).getMichSize();
  tools::parallel_for_each_CUDA(__count, [=] __device__(std::size_t index) {
    auto lorIndex = indexConverter.getLORIDFromRectangleID(__d_events[index].crystal1, __d_events[index].crystal2);
    if (lorIndex >= michSize)
      __out_factors[index] = 0;
    else
      __out_factors[index] = __michValues[lorIndex];
  });
}
void d_apply_correction_factor(
    float *d_values, std::span<core::MichStandardEvent> d_events, node::MichNormalization *michNorm,
    node::MichRandom *michRand, node::MichScatter *michScat, node::MichAttn *michAttn,
    cuda_sync_ptr<float> &d_addFactor, cuda_sync_ptr<float> &d_mulFactor) {
  int addFactorAssignTime = 0; // When assign first time, copy; otherwise, add
  int mulFactorAssignTime = 0; // When assign first time, copy; otherwise, multiply
  if (michNorm) {
    d_mulFactor.reserve(d_events.size());
    if (++mulFactorAssignTime == 1)
      example::d_parralel_copy(michNorm->getDNormFactorsBatch(d_events), d_mulFactor.data(), d_events.size());
    else
      example::d_parralel_mul(michNorm->getDNormFactorsBatch(d_events), d_mulFactor.data(), d_events.size());
  }
  if (michRand) {
    d_addFactor.reserve(d_events.size());
    if (++addFactorAssignTime == 1)
      example::d_parralel_copy(michRand->getDRandomFactorsBatch(d_events), d_addFactor.data(), d_events.size());
    else
      example::d_parralel_add(michRand->getDRandomFactorsBatch(d_events), d_addFactor.data(), d_events.size());
  }
  if (michScat) {
    d_addFactor.reserve(d_events.size());
    if (++addFactorAssignTime == 1)
      example::d_parralel_copy(michScat->getDScatterFactorsBatch(d_events), d_addFactor.data(), d_events.size());
    else
      example::d_parralel_add(michScat->getDScatterFactorsBatch(d_events), d_addFactor.data(), d_events.size());
  }
  if (michAttn) {
    d_mulFactor.reserve(d_events.size());
    if (++mulFactorAssignTime == 1)
      example::d_parralel_copy(michAttn->getDAttnFactorsBatch(d_events), d_mulFactor.data(), d_events.size());
    else
      example::d_parralel_mul(michAttn->getDAttnFactorsBatch(d_events), d_mulFactor.data(), d_events.size());
  }
  // if (d_addFactor)
  //   d_print_none_zero_average_value(d_addFactor.data(), d_events.size());
  tools::parallel_for_each_CUDA(d_events.size(), [=, d_addFactor = d_addFactor.data(),
                                                  d_mulFactor = d_mulFactor.data()] __device__(std::size_t index) {
    auto addFactor = d_addFactor ? d_addFactor[index] : 0.0f;
    auto mulFactor = d_mulFactor ? d_mulFactor[index] : 1.0f;
    if (mulFactor < 1e-8)
      d_values[index] = 0;
    else
      d_values[index] = d_values[index] + addFactor / mulFactor;
  });
}
void d_osem_fix_integral_value(
    std::span<float> __d_data) {
  if (__d_data.size() == 0)
    return;
  const auto maxValue = thrust::reduce(
      thrust::cuda::par.on(basic::cuda_ptr::default_stream()), thrust::device_pointer_cast(__d_data.data()),
      thrust::device_pointer_cast(__d_data.data() + __d_data.size()), 0.0f, thrust::maximum<float>());
  if (maxValue <= 0.0f)
    return;

  constexpr float ratio = 1e-7f;
  tools::parallel_for_each_CUDA(__d_data.size(), [=, __d_data = __d_data.data()] __device__(std::size_t idx) {
    auto &value = __d_data[idx];
    if (value < ratio * maxValue)
      value = 0;
    else
      value = 1.f / value;
  });
}
void d_osem_path_integral_batch_fov(
    core::TensorDataInput<float, 3> __img, std::span<core::MichStandardEvent const> __events, float __sample_rate,
    float *__out_values, cubef __roi) {
  tools::parallel_for_each_CUDA(__events.size(), [=, __events = __events.data()] __device__(std::size_t index) {
    __out_values[index] += node::impl::simple_path_integral(core::instant_random_float(index), __sample_rate, __img,
                                                            __events[index].geo1.O, __events[index].geo2.O, __roi);
  });
}
void d_osem_path_reverse_integral_batch_fov(
    core::TensorDataOutput<float, 3> __img, std::span<core::MichStandardEvent const> __events, float __sample_rate,
    cubef __roi) {
  tools::parallel_for_each_CUDA(__events.size(), [=, __events = __events.data()] __device__(std::size_t index) {
    float bias = static_cast<float>(core::instant_random_float(index));
    node::impl::simple_reverse_path_integral(bias, __sample_rate, __events[index].value, __img, __events[index].geo1.O,
                                             __events[index].geo2.O, __roi);
  });
}
void d_osem_path_integral_batch_TOF_fov(
    core::TensorDataInput<float, 3> __img, std::span<core::MichStandardEvent const> __events, float __sample_rate,
    float *__out_values, cubef __roi) {
  tools::parallel_for_each_CUDA(__events.size(), [=, __events = __events.data()] __device__(std::size_t index) {
    core::Vector<int16_t, 2> pair_tofMean =
        core::Vector<int16_t, 2>::create(__events[index].cry1_tof_mean, __events[index].cry2_tof_mean);
    core::Vector<int16_t, 2> pair_tofDev =
        core::Vector<int16_t, 2>::create(__events[index].cry1_tof_deviation, __events[index].cry2_tof_deviation);
    float bias = static_cast<float>(core::instant_random_float(index));
    __out_values[index] +=
        node::impl::simple_path_integral_TOF(bias, __sample_rate, __img, __events[index].geo1.O, __events[index].geo2.O,
                                             pair_tofMean, pair_tofDev, __events[index].tof, __roi);
  });
}
void d_osem_path_reverse_integral_batch_TOF_fov(
    core::TensorDataOutput<float, 3> __img, std::span<core::MichStandardEvent const> __events, float __sample_rate,
    cubef __roi) {
  tools::parallel_for_each_CUDA(__events.size(), [=, __events = __events.data()] __device__(std::size_t index) {
    core::Vector<int16_t, 2> pair_tofMean =
        core::Vector<int16_t, 2>::create(__events[index].cry1_tof_mean, __events[index].cry2_tof_mean);
    core::Vector<int16_t, 2> pair_tofDev =
        core::Vector<int16_t, 2>::create(__events[index].cry1_tof_deviation, __events[index].cry2_tof_deviation);
    float bias = static_cast<float>(core::instant_random_float(index));
    node::impl::simple_reverse_path_integral_TOF(bias, __sample_rate, __events[index].value, __img,
                                                 __events[index].geo1.O, __events[index].geo2.O, pair_tofMean,
                                                 pair_tofDev, __events[index].tof, __roi);
  });
}
std::vector<cubef> select_roi_by_8_pars(
    cubef __roi) {
  auto begin = __roi[0];
  auto size = (__roi[1] - __roi[0]) / 2;
  std::vector<cubef> rois;
  for (const auto index : core::MDSpan<3>::create(2, 2, 2)) {
    auto newBegin = begin + size * index.to<float>();
    auto newEnd = newBegin + size;
    rois.push_back(cubef::create(newBegin, newEnd));
  }
  return rois;
}
#define USE_CACHE_OPTIMIZE 1
#define CACHE_OPTIMIZE_THRESHOLD (48 * 1024 * 1024) // 48MB
void d_simple_path_integral_batch(
    core::TensorDataInput<float, 3> __img, std::span<core::MichStandardEvent const> __events, float __sample_rate,
    float *__out_values) {
#if USE_CACHE_OPTIMIZE
  const auto cacheOptimize = __img.grid.totalSize() * sizeof(float) > CACHE_OPTIMIZE_THRESHOLD;
#else
  const auto cacheOptimize = false;
#endif
  const auto rois =
      cacheOptimize ? select_roi_by_8_pars(__img.grid.bounding_box()) : std::vector<cubef>{__img.grid.bounding_box()};
  cudaMemsetAsync(__out_values, 0, sizeof(float) * __events.size(), basic::cuda_ptr::default_stream());
  for (const auto &roi : rois)
    d_osem_path_integral_batch_fov(__img, __events, __sample_rate, __out_values, roi);
}
void d_simple_path_reverse_integral_batch(
    core::TensorDataOutput<float, 3> __img, std::span<core::MichStandardEvent const> __events, float __sample_rate) {
#if USE_CACHE_OPTIMIZE
  const auto cacheOptimize = __img.grid.totalSize() * sizeof(float) > CACHE_OPTIMIZE_THRESHOLD;
#else
  const auto cacheOptimize = false;
#endif
  const auto rois =
      cacheOptimize ? select_roi_by_8_pars(__img.grid.bounding_box()) : std::vector<cubef>{__img.grid.bounding_box()};
  for (const auto &roi : rois)
    d_osem_path_reverse_integral_batch_fov(__img, __events, __sample_rate, roi);
}
void d_simple_path_integral_batch_TOF(
    core::TensorDataInput<float, 3> __img, std::span<core::MichStandardEvent const> __events, float __sample_rate,
    float *__out_values) {
#if USE_CACHE_OPTIMIZE
  const auto cacheOptimize = __img.grid.totalSize() * sizeof(float) > CACHE_OPTIMIZE_THRESHOLD;
#else
  const auto cacheOptimize = false;
#endif
  const auto rois =
      cacheOptimize ? select_roi_by_8_pars(__img.grid.bounding_box()) : std::vector<cubef>{__img.grid.bounding_box()};
  cudaMemsetAsync(__out_values, 0, sizeof(float) * __events.size(), basic::cuda_ptr::default_stream());
  for (const auto &roi : rois)
    d_osem_path_integral_batch_TOF_fov(__img, __events, __sample_rate, __out_values, roi);
}
void d_simple_path_reverse_integral_batch_TOF(
    core::TensorDataOutput<float, 3> __img, std::span<core::MichStandardEvent const> __events, float __sample_rate) {
#if USE_CACHE_OPTIMIZE
  const auto cacheOptimize = __img.grid.totalSize() * sizeof(float) > CACHE_OPTIMIZE_THRESHOLD;
#else
  const auto cacheOptimize = false;
#endif
  const auto rois =
      cacheOptimize ? select_roi_by_8_pars(__img.grid.bounding_box()) : std::vector<cubef>{__img.grid.bounding_box()};
  for (const auto &roi : rois)
    d_osem_path_reverse_integral_batch_TOF_fov(__img, __events, __sample_rate, roi);
}
#undef USE_CACHE_OPTIMIZE

float d_cal_update_measurements(
    float const *__updateImage, float const *__senmapImage, std::size_t count) {
  if (count == 0)
    return 0.0f;
  float sum = thrust::transform_reduce(
      thrust::cuda::par.on(basic::cuda_ptr::default_stream()), thrust::counting_iterator<std::size_t>(0),
      thrust::counting_iterator<std::size_t>(count),
      [=] __device__(std::size_t idx) -> float {
        if (__updateImage[idx] > 0 && __senmapImage[idx] > 0)
          return core::FMath<float>::abs(core::FMath<float>::flog(__updateImage[idx] / __senmapImage[idx]));
        else
          return 0.f;
      },
      0.0, thrust::plus<>());
  return sum / count;
}
void d_count_from_listmode(
    std::span<basic::Listmode_t const> __listmodes, std::size_t const *__lorids, std::size_t __lorBegin,
    core::MichDefine __michDefine, float *__out_counts) {
  auto indexConverter = core::IndexConverter::create(__michDefine);
  auto michSize = core::MichInfoHub::create(__michDefine).getMichSize();
  tools::parallel_for_each_CUDA(
      __listmodes.size(), [=, __listmodes = __listmodes.data()] __device__(std::size_t index) {
        auto uniformID1 = core::mich::getUniformIdFromFlatId(__michDefine.polygon, __michDefine.detector,
                                                             static_cast<int>(__listmodes[index].globalCrystalIndex1));
        auto uniformID2 = core::mich::getUniformIdFromFlatId(__michDefine.polygon, __michDefine.detector,
                                                             static_cast<int>(__listmodes[index].globalCrystalIndex2));
        auto rid1 = core::mich::getRectangleIDFromUniformID(__michDefine.polygon, __michDefine.detector, uniformID1);
        auto rid2 = core::mich::getRectangleIDFromUniformID(__michDefine.polygon, __michDefine.detector, uniformID2);
        auto lorId = indexConverter.getLORIDFromRectangleID(rid1, rid2);
        if (lorId < michSize)
          atomicAdd(&__out_counts[__lorids[lorId]], 1.0f);
      });
}
void d_redirect_from_mich_from_slice_range(
    std::size_t __sliceBegin, std::size_t __sliceEnd, float const *__d_michValues, float *__d_outValues,
    core::MichDefine __michDefine) {
  auto indexConverter = core::IndexConverter::create(__michDefine);
  auto michInfo = core::MichInfoHub::create(__michDefine);
  auto binNum = michInfo.getBinNum();
  auto viewNum = michInfo.getViewNum();
  tools::parallel_for_each_CUDA((__sliceEnd - __sliceBegin) * binNum * viewNum, [=] __device__(std::size_t index) {
    auto slice = index / (binNum * viewNum) + __sliceBegin;
    auto lorInSl = index % (binNum * viewNum);
    auto lorId = slice * (binNum * viewNum) + lorInSl;
    auto michSize = michInfo.getMichSize();
    if (lorId < michSize)
      __d_outValues[index] = __d_michValues[lorId];
    else
      __d_outValues[index] = 0.0f;
  });
}
void d_redirect_from_mich_from_slice_range(
    std::size_t __sliceBegin, std::size_t __sliceEnd, std::span<basic::Listmode_t const> __d_listmodes,
    float *__d_outValues, core::MichDefine __michDefine) {
  auto indexConverter = core::IndexConverter::create(__michDefine);
  auto michInfo = core::MichInfoHub::create(__michDefine);
  auto binNum = michInfo.getBinNum();
  auto viewNum = michInfo.getViewNum();
  tools::parallel_for_each_CUDA(
      __d_listmodes.size(), [=, __d_listmodes = __d_listmodes.data()] __device__(std::size_t index) {
        auto lm = __d_listmodes[index];
        auto uniformID1 = core::mich::getUniformIdFromFlatId(__michDefine.polygon, __michDefine.detector,
                                                             static_cast<int>(lm.globalCrystalIndex1));
        auto uniformID2 = core::mich::getUniformIdFromFlatId(__michDefine.polygon, __michDefine.detector,
                                                             static_cast<int>(lm.globalCrystalIndex2));
        auto rid1 = core::mich::getRectangleIDFromUniformID(__michDefine.polygon, __michDefine.detector, uniformID1);
        auto rid2 = core::mich::getRectangleIDFromUniformID(__michDefine.polygon, __michDefine.detector, uniformID2);
        auto lorId = indexConverter.getLORIDFromRectangleID(rid1, rid2);
        if (lorId >= michInfo.getMichSize())
          return; // Out of range

        auto lorBegin = __sliceBegin * binNum * viewNum;
        auto lorEnd = __sliceEnd * binNum * viewNum;
        if (lorId < lorBegin || lorId >= lorEnd)
          return;                                          // Out of range
        atomicAdd(&__d_outValues[lorId - lorBegin], 1.0f); // Listmode events have implicit value = 1
      });
}
} // namespace openpni::experimental::node::impl