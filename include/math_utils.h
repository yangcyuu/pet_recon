#pragma once

#include "define.h"
#include "utils.h"

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> line_in_grid(const torch::Tensor &start,
                                                                     const torch::Tensor &end, const Grids<3> &grids,
                                                                     const Vector3f &extend_size = {0.0f, 0.0f, 0.0f});

template<typename T>
torch::Tensor gaussian_sample(torch::Tensor rd, const torch::Tensor &sigma, const T &mean) {
  rd = torch::clamp(rd, 1e-6f, 1.0f - 1e-6f);
  if constexpr (std::is_fundamental_v<T>) {
    return torch::erfinv(2 * rd - 1) * sigma * std::sqrt(2.0f) + mean;
  } else if constexpr (std::is_same_v<T, torch::Tensor>) {
    return torch::erfinv(2 * rd - 1) * sigma * std::sqrt(2.0f) + mean.to(sigma.device());
  } else {
    static_assert(false, "Unsupported type for mean in gaussian_sample");
    return torch::Tensor();
  }
}
