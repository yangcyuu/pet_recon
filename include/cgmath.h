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

inline bool lor_in_image(const openpni::experimental::core::Vector<float, 3> &p0,
                         const openpni::experimental::core::Vector<float, 3> &p1,
                         const openpni::experimental::core::Vector<float, 3> &voxel_size,
                         const openpni::experimental::core::Vector<float, 3> &image_size) {
  openpni::experimental::core::Vector<float, 3> half_image_size = image_size * 0.5f * voxel_size;
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
        return false;
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
        return false;
      }
    }
  }

  return (tmin <= tmax) && (tmax >= 0.0f) && (tmin <= 1.0f);
}
