#pragma once

#include <Polygon.hpp>
#include "define.h"
#include "utils.h"

template<typename T>
T gamma_to_linear(T value, T gamma = static_cast<T>(2.2)) {
  static std::array<T, 256> LUT = [&] {
    std::array<T, 256> lut = {};
    for (auto i = 0; i < 256; ++i) {
      lut[i] = std::pow(static_cast<T>(i) / 255, gamma);
    }
    return lut;
  }();

  return LUT[std::clamp<size_t>(std::round(value * 255), 0, 255)];
}

template<typename T>
T linear_to_gamma(T value, T gamma = static_cast<T>(2.2)) {
  static std::array<T, 1024> LUT = [&] {
    std::array<T, 1024> lut = {};
    for (auto i = 0; i < 1024; ++i) {
      lut[i] = std::clamp<T>(255 * std::pow(static_cast<T>(i) / 1023, 1 / gamma) + static_cast<T>(0.5), 0, 255);
    }
    return lut;
  }();
  return LUT[std::clamp<size_t>(std::round(value * 1023), 0, 1023)];
}

inline std::optional<std::pair<float, float>>
lor_in_image(const openpni::experimental::core::Vector<float, 3> &p0,
             const openpni::experimental::core::Vector<float, 3> &p1,
             const openpni::experimental::core::Vector<float, 3> &voxel_size,
             const openpni::experimental::core::Vector<float, 3> &image_size,
             const openpni::experimental::core::Vector<float, 3> &extend_size = {0, 0, 0}) {
  openpni::experimental::core::Vector<float, 3> half_image_size = image_size * 0.5f * voxel_size + extend_size;
  openpni::experimental::core::Vector<float, 3> min_bound = -half_image_size;
  openpni::experimental::core::Vector<float, 3> max_bound = half_image_size;

  openpni::experimental::core::Vector<float, 3> dir = p1 - p0;
  float tmin = 0.0f;
  float tmax = 1.0f;

  float epsilon = std::max({half_image_size[0], half_image_size[1], half_image_size[2]}) * 1e-6f;

  for (int i = 0; i < 3; ++i) {
    float origin = p0[i];
    float direction = dir[i];
    float min_b = min_bound[i];
    float max_b = max_bound[i];


    if (std::abs(direction) < epsilon) {
      if (origin < min_b || origin > max_b) {
        return std::nullopt;
      }
    } else {
      float t1 = (min_b - origin) / direction;
      float t2 = (max_b - origin) / direction;

      if (t1 > t2) {
        std::swap(t1, t2);
      }

      tmin = std::max(tmin, t1);
      tmax = std::min(tmax, t2);

      if (tmin > tmax) {
        return std::nullopt;
      }
    }
  }

  if ((tmin <= tmax) && (tmax >= 0.0f) && (tmin <= 1.0f)) {
    return std::make_pair(tmin, tmax);
  }
  return std::nullopt;
}

inline std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> clip_lors(const torch::Tensor &p0,
                                                                         const torch::Tensor &p1,
                                                                         const torch::Tensor &voxel_size,
                                                                         const torch::Tensor &image_size) {
  // p0, p1: [N, N_crystal, 3]
  // voxel_size: [3]
  // image_size: [3]
  torch::Tensor image_min = -0.5f * voxel_size.unsqueeze(0).unsqueeze(0) * image_size.unsqueeze(0).unsqueeze(0);
  torch::Tensor image_max = 0.5f * voxel_size.unsqueeze(0).unsqueeze(0) * image_size.unsqueeze(0).unsqueeze(0);

  constexpr float epsilon = 1e-6f;
  constexpr float largest = 1e6f;
  torch::Tensor direction = p1 - p0;
  torch::Tensor abs_direction = torch::abs(direction);
  // torch::Tensor t0 = (image_min - p0) / (direction + 1e-8f);
  // torch::Tensor t1 = (image_max - p0) / (direction + 1e-8f);
  torch::Tensor t0 =
      torch::where(abs_direction > epsilon, (image_min - p0) / direction, torch::full_like(direction, -largest));
  torch::Tensor t1 =
      torch::where(abs_direction > epsilon, (image_max - p0) / direction, torch::full_like(direction, largest));

  torch::Tensor tmin = std::get<0>(torch::min(t0, t1).max(-1)); // [N, N_crystal]
  torch::Tensor tmax = std::get<0>(torch::max(t0, t1).min(-1));

  torch::Tensor valid_mask = (tmin < tmax) & (tmax > 0.0f) & (tmin < 1.0f);

  tmin = torch::clamp(tmin.unsqueeze(-1), 0.0f, 1.0f); // [N, N_crystal, 1]
  tmax = torch::clamp(tmax.unsqueeze(-1), 0.0f, 1.0f);

  torch::Tensor clipped_p0 = p0 + direction * tmin;
  torch::Tensor clipped_p1 = p0 + direction * tmax;

  return std::make_tuple(clipped_p0, clipped_p1, valid_mask);
}
