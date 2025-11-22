#pragma once

#include <cmath>
#include <concepts>
#include <iostream>

#include <torch/torch.h>
#include <tuple>


#define ERROR_AND_EXIT(msg, ...)                                                                                       \
  do {                                                                                                                 \
    std::cerr << std::format("{}:{}: {}", __FILE__, __LINE__, std::format(msg, ##__VA_ARGS__)) << std::endl;           \
    std::exit(EXIT_FAILURE);                                                                                           \
  } while (false)

#define CHECK_TENSOR_NAN(tensor, msg, ...)                                                                             \
  do {                                                                                                                 \
    torch::NoGradGuard no_grad;                                                                                        \
    if (torch::isnan(tensor).any().item<bool>()) {                                                                     \
      ERROR_AND_EXIT(msg, ##__VA_ARGS__);                                                                              \
    }                                                                                                                  \
  } while (false)

#define CHECK_TENSOR_INF(tensor, msg, ...)                                                                             \
  do {                                                                                                                 \
    torch::NoGradGuard no_grad;                                                                                        \
    if (torch::isinf(tensor).any().item<bool>()) {                                                                     \
      ERROR_AND_EXIT(msg, ##__VA_ARGS__);                                                                              \
    }                                                                                                                  \
  } while (false)

#define MARK_AS_UNUSED(x) static_cast<void>(x)
