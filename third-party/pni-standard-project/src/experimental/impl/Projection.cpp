#include "Projection.h"

#include "include/experimental/tools/Parallel.hpp"
namespace openpni::experimental::node::impl {
void h_gen_some_attn_factors(
    float *__out_factors, std::span<core::MichStandardEvent const> __events, const float *__attn_map,
    core::Grids<3> __map) {
  tools::parallel_for_each(__events.size(), [&](std::size_t index) {
    __out_factors[index] =
        instant_path_integral(core::instant_random_float(index), core::TensorDataInput<float, 3>{__map, __attn_map},
                              __events[index].geo1.O, __events[index].geo2.O);
    __out_factors[index] = core::FMath<float>::fexp(-__out_factors[index]);
  });
}
float h_cal_update_measurements(
    float const *h_updateImage, float const *h_senmapImage, std::size_t count) {
  if (count == 0)
    return 0;
  double sum = 0;
  for (const auto i : std::views::iota(0ull, count))
    if (h_updateImage[i] > 0 && h_senmapImage[i] > 0)
      sum += std::abs(std::log(h_updateImage[i] / h_senmapImage[i]));
  return sum / count;
}
void h_apply_correction_factor(
    std::span<float> values, std::span<core::MichStandardEvent> _event, node::MichNormalization *michNorm,
    node::MichRandom *michRand, node::MichScatter *michScat) {
  std::vector<float> addFactor(values.size(), 0.0f);
  std::vector<float> multFactor(values.size(), 1.0f);
  if (michNorm) {
    auto normFactors =
        michNorm->getHNormFactorsBatch(std::span<core::MichStandardEvent const>(_event.data(), values.size()));
    tools::parallel_for_each(values.size(), [&](std::size_t index) { multFactor[index] = normFactors[index]; });
  }
  if (michRand) {
    auto randFactors =
        michRand->getHRandomFactorsBatch(std::span<core::MichStandardEvent const>(_event.data(), values.size()));
    tools::parallel_for_each(values.size(), [&](std::size_t index) { addFactor[index] += randFactors[index]; });
  }
  if (michScat) {
    auto scatFactors =
        michScat->getHScatterFactorsBatch(std::span<core::MichStandardEvent const>(_event.data(), values.size()));
    tools::parallel_for_each(values.size(), [&](std::size_t index) { addFactor[index] += scatFactors[index]; });
  }
  tools::parallel_for_each(values.size(), [=](std::size_t index) {
    if (multFactor[index] < 1e-8)
      values[index] = 0;
    else
      values[index] = values[index] + addFactor[index] / multFactor[index];
  });
}

} // namespace openpni::experimental::node::impl
