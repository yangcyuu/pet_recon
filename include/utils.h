#pragma once

#include <cmath>
#include <concepts>
#include <iostream>
#include <opencv2/opencv.hpp>

#include <torch/torch.h>
#include <tuple>


#define ERROR_AND_EXIT(msg, ...)                                                                                       \
  do {                                                                                                                 \
    std::println(std::cerr, msg, ##__VA_ARGS__);                                                                       \
    std::exit(EXIT_FAILURE);                                                                                           \
  } while (false)

#define MARK_AS_UNUSED(x) static_cast<void>(x)

template<typename T>
concept array_accessible = requires(T a, size_t i) {
  { a[i] };
};

inline torch::Tensor gaussian_sample(torch::Tensor rd, const torch::Tensor &sigma,
                                     const torch::Tensor &mean = torch::zeros(1, torch::kCUDA)) {
  // rd = torch::clamp(rd, torch::nextafter(torch::zeros(1, rd.device()), torch::ones(1, rd.device())),
  //                   torch::nextafter(torch::ones(1, rd.device()), torch::zeros(1, rd.device())));
  rd = torch::clamp(rd, 1e-6f, 1.0f - 1e-6f);
  return torch::erfinv(2 * rd - 1) * sigma * std::sqrt(2.0f) + mean;
}
