#pragma once

#include "utils.h"

template <typename T>
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

template <typename T>
T linear_to_gamma(T value, T gamma = static_cast<T>(2.2)) {
  static std::array<T, 1024> LUT = [&] {
    std::array<T, 1024> lut = {};
    for (auto i = 0; i < 1024; ++i) {
      lut[i] =
          std::clamp<T>(255 * std::pow(static_cast<T>(i) / 1023, 1 / gamma) +
                            static_cast<T>(0.5),
                        0, 255);
    }
    return lut;
  }();
  return LUT[std::clamp<size_t>(std::round(value * 1023), 0, 1023)];
}