#pragma once
#include <algorithm>
#include <cinttypes>
#include <vector>

#include "../core/Vector.hpp"
#include "Parallel.cuh"
#include "Parallel.hpp"
namespace openpni::experimental::tools {

struct _ChunckedRange {
  std::vector<std::pair<int64_t, int64_t>> by_max_size(
      int64_t start, int64_t end, int64_t max_size) const {
    if (max_size <= 0)
      return {};
    if (start >= end)
      return {};
    std::vector<std::pair<int64_t, int64_t>> result;
    for (int64_t i = start; i < end; i += max_size) {
      result.emplace_back(i, std::min(i + max_size, end));
    }
    return result;
  }
  std::vector<std::pair<int64_t, int64_t>> by_group_count(
      int64_t start, int64_t end, int64_t group_count) const {
    if (group_count <= 0)
      return {};
    if (start >= end)
      return {};
    int64_t total_size = end - start;
    int64_t group_size = (total_size + group_count - 1) / group_count;
    return by_max_size(start, end, group_size);
  }
  std::vector<std::pair<int64_t, int64_t>> by_balanced_max_size(
      int64_t start, int64_t end, int64_t max_size, float balance_factor = 0.3) const {
    if (max_size <= 0)
      return {};
    if (start >= end)
      return {};
    balance_factor = std::clamp(balance_factor, 0.0f, 1.0f);
    int64_t total_size = end - start;
    int64_t unbalanced_count = total_size % max_size;
    if (unbalanced_count == 0)
      return by_max_size(start, end, max_size);
    if (unbalanced_count >= balance_factor * max_size)
      return by_max_size(start, end, max_size);
    return by_group_count(start, end, total_size / max_size + 1);
  }
};
constexpr inline _ChunckedRange chunked_ranges_generator{};

template <typename T>
inline std::vector<T> make_stepped_vector(
    T start, T end, T step) {
  std::vector<T> result;
  for (; start < end; start += step)
    result.push_back(start);
  return result;
}

struct _ContinuousRange {
  std::vector<core::Vector<int64_t, 2>> by_max_difference_ordered(
      int64_t start, int64_t end, int64_t max_difference) const {
    if (max_difference <= 0)
      return {};
    if (start >= end)
      return {};
    std::vector<core::Vector<int64_t, 2>> result;
    core::MDBeginEndSpan<2> span = core::MDBeginEndSpan<2>::create(core::Vector<int64_t, 2>::create(start, start),
                                                                   core::Vector<int64_t, 2>::create(end, end));
    for (const auto &[a, b] : span)
      if (std::abs(a - b) < max_difference)
        result.push_back(core::Vector<int64_t, 2>::create(a, b));
    return result;
  }
  std::vector<core::Vector<int64_t, 2>> by_max_difference_unordered(
      int64_t start, int64_t end, int64_t max_difference) const {
    if (max_difference <= 0)
      return {};
    if (start >= end)
      return {};
    std::vector<core::Vector<int64_t, 2>> result;
    core::MDBeginEndSpan<2> span = core::MDBeginEndSpan<2>::create(core::Vector<int64_t, 2>::create(start, start),
                                                                   core::Vector<int64_t, 2>::create(end, end));
    for (const auto &[a, b] : span)
      if (std::abs(a - b) < max_difference && a <= b)
        result.push_back(core::Vector<int64_t, 2>::create(a, b));
    return result;
  }
};
constexpr inline _ContinuousRange continuous_range_generator{};

} // namespace openpni::experimental::tools
