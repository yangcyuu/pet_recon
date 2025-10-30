#pragma once
#include <ranges>
#include <vector>
namespace openpni::experimental::tools {
inline auto to_vector(
    auto &&range) {
  return std::vector(range.begin(), range.end());
}
inline std::vector<std::size_t> fill_vector(
    std::size_t begin, std::size_t end) {
  std::vector<std::size_t> result;
  result.reserve(end - begin);
  for (std::size_t i = begin; i < end; i++)
    result.push_back(i);
  return result;
}
inline auto find_first_if_or(
    auto &&range, auto &&pred, auto &&default_value) {
  auto it = std::ranges::find_if(range, pred);
  if (it != range.end())
    return *it;
  else
    return default_value;
}
} // namespace openpni::experimental::tools
