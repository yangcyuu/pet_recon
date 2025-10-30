#include "include/experimental/core/Matrix.hpp"

#include <gtest/gtest.h>
#include <type_traits>

#include "include/experimental/core/Vector.hpp"

#define TEST_SUITE_NAME openpni_experimental_core_Matrix
#define test(name) TEST(TEST_SUITE_NAME, name)

using openpni::experimental::core::Matrix;
using openpni::experimental::core::Vector;

test(
    初始化与元素访问) {
  auto m = Matrix<int, 2, 3>::create();
  int value = 0;
  for (int r = 0; r < 2; r++)
    for (int c = 0; c < 3; c++)
      m(r, c) = ++value;

  EXPECT_EQ(m(0, 0), 1);
  EXPECT_EQ(m(0, 1), 2);
  EXPECT_EQ(m(0, 2), 3);
  EXPECT_EQ(m(1, 0), 4);
  EXPECT_EQ(m(1, 1), 5);
  EXPECT_EQ(m(1, 2), 6);

  const auto &cm = m;
  EXPECT_EQ(cm(1, 1), 5);
}

test(
    行列抽取) {
  auto m = Matrix<int, 3, 3>::create();
  int value = 0;
  for (int r = 0; r < 3; r++)
    for (int c = 0; c < 3; c++)
      m(r, c) = value++;

  Vector<int, 3> row = m.take_row(1);
  EXPECT_EQ(row[0], 3);
  EXPECT_EQ(row[1], 4);
  EXPECT_EQ(row[2], 5);

  Vector<int, 3> column = m.take_column(2);
  EXPECT_EQ(column[0], 2);
  EXPECT_EQ(column[1], 5);
  EXPECT_EQ(column[2], 8);
}

test(
    矩阵与向量乘法) {
  auto m = Matrix<int, 2, 3>::create();
  int value = 1;
  for (int r = 0; r < 2; r++)
    for (int c = 0; c < 3; c++)
      m(r, c) = value++;

  Vector<double, 3> v = Vector<double, 3>::create(0.5, 1.0, 1.5);
  using ResultType = decltype(m * v);
  EXPECT_TRUE((std::is_same_v<ResultType, Vector<double, 2>>));
  ResultType result = m * v;

  EXPECT_NEAR(result[0], 1 * 0.5 + 2 * 1.0 + 3 * 1.5, 1e-9);
  EXPECT_NEAR(result[1], 4 * 0.5 + 5 * 1.0 + 6 * 1.5, 1e-9);

  Vector<int, 3> vi = Vector<int, 3>::create(1, 2, 3);
  auto int_result = m * vi;
  EXPECT_EQ(int_result[0], 14);
  EXPECT_EQ(int_result[1], 32);
}

test(
    矩阵与矩阵乘法) {
  auto a = Matrix<int, 2, 3>::create();
  int value = 1;
  for (int r = 0; r < 2; r++)
    for (int c = 0; c < 3; c++)
      a(r, c) = value++;

  auto b = Matrix<float, 3, 2>::create();
  float fvalue = 1.0f;
  for (int r = 0; r < 3; r++)
    for (int c = 0; c < 2; c++)
      b(r, c) = fvalue++;

  using ResultType = decltype(a * b);
  EXPECT_TRUE((std::is_same_v<ResultType, Matrix<float, 2, 2>>));
  ResultType result = a * b;

  EXPECT_FLOAT_EQ(result(0, 0), 1 * 1.0f + 2 * 3.0f + 3 * 5.0f);
  EXPECT_FLOAT_EQ(result(0, 1), 1 * 2.0f + 2 * 4.0f + 3 * 6.0f);
  EXPECT_FLOAT_EQ(result(1, 0), 4 * 1.0f + 5 * 3.0f + 6 * 5.0f);
  EXPECT_FLOAT_EQ(result(1, 1), 4 * 2.0f + 5 * 4.0f + 6 * 6.0f);
}
