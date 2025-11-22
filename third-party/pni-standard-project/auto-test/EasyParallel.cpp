// #include "include/experimental/example/EasyParallel.hpp"

// #include <gtest/gtest.h>
// #include <type_traits>

// #include "include/experimental/tools/Parallel.hpp"

// #define TEST_SUITE_NAME openpni_experimental_example_EasyParallel
// #define test(name) TEST(TEST_SUITE_NAME, name)
// #define SET_PARALLEL \
//   { \
//     openpni::experimental::tools::cpu_threads() \
//         .setThreadNumType(openpni::experimental::tools::MAX_THREAD) \
//         .setScheduleType(openpni::experimental::tools::DYNAMIC) \
//         .setScheduleNum(64); \
//   }

// using namespace openpni::experimental::example;

// template <typename T>
// std::vector<T> generate_sequence(
//     std::size_t size, std::function<T(std::size_t)> generator) {
//   std::vector<T> vec;
//   vec.reserve(size);
//   for (std::size_t i = 0; i < size; ++i) {
//     vec.push_back(generator(i));
//   }
//   return vec;
// }
// template <typename T>
// bool expected_equal(
//     std::vector<T> const &data, std::vector<T> const &expected) {
//   if (data.size() != expected.size())
//     return false;
//   for (std::size_t i = 0; i < data.size(); ++i) {
//     if (data[i] != expected[i])
//       return false;
//   }
//   return true;
// }

// test(
//     并行加_CPU) {
//   SET_PARALLEL;
//   constexpr std::size_t dataSize = 1'000'000;
//   auto dataA = generate_sequence<float>(dataSize, [](std::size_t index) { return static_cast<float>(index * index);
//   }); auto dataB = generate_sequence<float>(dataSize, [](std::size_t index) { return static_cast<float>(index + 3);
//   }); std::vector<float> result(dataSize, 0); h_parallel_add(dataA.data(), dataB.data(), result.data(), dataSize);
//   auto expected = generate_sequence<float>(
//       dataSize, [](std::size_t index) { return static_cast<float>(index * index + (index + 3)); });
//   EXPECT_TRUE(expected_equal(result, expected));
// }
