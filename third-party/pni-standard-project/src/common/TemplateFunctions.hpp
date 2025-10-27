#pragma once
#include <cstring>
#include <memory>
#include <optional>

namespace openpni::common {
template <typename Map>
typename Map::mapped_type map_or(const Map &__m, typename Map::key_type __v,
                                 typename Map::mapped_type &&__or) {
  if (const auto iter = __m.find(__v); iter == __m.end())
    return __or;
  else
    return iter->second;
}
template <typename Map>
std::optional<typename Map::mapped_type> map_optional(const Map &__m,
                                                      typename Map::key_type __v) {
  if (const auto iter = __m.find(__v); iter == __m.end())
    return std::nullopt;
  else
    return iter->second;
}
template <typename Map, typename Func>
std::optional<typename Map::mapped_type> map_find_if(const Map &__m, Func &&__func) {
  for (const auto &[key, value] : __m) {
    if (__func(key, value))
      return value;
  }
  return std::nullopt;
}

template <typename T>
bool is_one_of(const T &__value, const std::initializer_list<T> &__list) noexcept {
  for (const auto &item : __list)
    if (__value == item)
      return true;
  return false;
}

template <typename T, int BytesOutput>
std::unique_ptr<char[]> cut_to_bytes(const T *begin, const T *end) noexcept {
  static_assert(BytesOutput > 0, "BytesOutput must be greater than 0");
  static_assert(BytesOutput < sizeof(T), "BytesOutput must be less than sizeof(T)");
  const auto elements = end - begin;
  const auto bytes = elements * BytesOutput;

  auto result = std::make_unique<char[]>(bytes);
  for (std::size_t i = 0; i < elements; ++i) {
    // 将输入元素的低 BytesOutput 字节复制到输出数组中
    ::memcpy(result.get() + i * BytesOutput, reinterpret_cast<const char *>(begin + i),
             BytesOutput);
  }
  return result;
}

template <typename T>
void make_unique_uninitialized(std::unique_ptr<T[]> &ptr, std::size_t count) noexcept {
  static_assert(std::is_trivially_default_constructible_v<T>,
                "T must be trivially default constructible");
  ptr = std::make_unique_for_overwrite<T[]>(count);
}

template <typename T>
void make_unique_memset(std::unique_ptr<T[]> &ptr, std::size_t count) noexcept {
  static_assert(std::is_trivially_default_constructible_v<T>,
                "T must be trivially default constructible");
  ptr = std::make_unique_for_overwrite<T[]>(count);
  ::memset(ptr.get(), 0, sizeof(T) * count);
}

} // namespace openpni::common
