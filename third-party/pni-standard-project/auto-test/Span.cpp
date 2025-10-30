#include "include/experimental/core/Span.hpp"

#include <gtest/gtest.h>
#define TEST_SUITE_NAME openpni_experimental_core_Span
#define test(name) TEST(TEST_SUITE_NAME, name)
using namespace openpni::experimental::core;

test(
    一维范围测试) {
  MDSpan<1> span = MDSpan<1>::create(10);
  EXPECT_EQ(span.totalSize(), 10);
  EXPECT_TRUE(span.inBounds(Vector<int64_t, 1>::create(0)));
  EXPECT_TRUE(span.inBounds(Vector<int64_t, 1>::create(9)));
  EXPECT_FALSE(span.inBounds(Vector<int64_t, 1>::create(-1)));
  EXPECT_FALSE(span.inBounds(Vector<int64_t, 1>::create(10)));
  EXPECT_EQ(span(Vector<int64_t, 1>::create(0)), 0);
  EXPECT_EQ(span(Vector<int64_t, 1>::create(9)), 9);
  EXPECT_EQ((span[Vector<int64_t, 1>::create(0)]), 0);
  EXPECT_EQ((span[Vector<int64_t, 1>::create(9)]), 9);
  for (int64_t i = 0; i < span.totalSize(); i++) {
    Vector<int64_t, 1> index = span.toIndex(i);
    EXPECT_EQ(index[0], i);
    EXPECT_EQ(span(index), i);
  }
}

test(
    二维范围测试) {
  MDSpan<2> span = MDSpan<2>::create(3, 4);
  EXPECT_EQ(span.totalSize(), 12);
  EXPECT_TRUE(span.inBounds(Vector<int64_t, 2>::create(0, 0)));
  EXPECT_TRUE(span.inBounds(Vector<int64_t, 2>::create(2, 3)));
  EXPECT_FALSE(span.inBounds(Vector<int64_t, 2>::create(-1, 0)));
  EXPECT_FALSE(span.inBounds(Vector<int64_t, 2>::create(0, -1)));
  EXPECT_FALSE(span.inBounds(Vector<int64_t, 2>::create(3, 0)));
  EXPECT_FALSE(span.inBounds(Vector<int64_t, 2>::create(0, 4)));
  EXPECT_EQ(span(Vector<int64_t, 2>::create(0, 0)), 0);
  EXPECT_EQ(span(Vector<int64_t, 2>::create(0, 1)), 3);
  EXPECT_EQ(span(Vector<int64_t, 2>::create(1, 0)), 1);
  EXPECT_EQ(span(Vector<int64_t, 2>::create(2, 3)), 11);
  EXPECT_EQ((span[Vector<int64_t, 2>::create(0, 0)]), 0);
  EXPECT_EQ((span[Vector<int64_t, 2>::create(0, 1)]), 3);
  EXPECT_EQ((span[Vector<int64_t, 2>::create(1, 0)]), 1);
  EXPECT_EQ((span[Vector<int64_t, 2>::create(2, 3)]), 11);
  for (int64_t i = 0; i < span.totalSize(); i++) {
    Vector<int64_t, 2> index = span.toIndex(i);
    EXPECT_EQ(span(index), i);
    EXPECT_EQ((span[index]), i);
    EXPECT_EQ((index[0] + index[1] * span.dimSize[0]), i);
    EXPECT_EQ((index[1]), i / span.dimSize[0]);
    EXPECT_EQ((index[0]), i % span.dimSize[0]);
    EXPECT_TRUE(span.inBounds(index));
    for (int j = -1; j <= 1; j++)
      for (int k = -1; k <= 1; k++) {
        if (j == 0 && k == 0)
          continue;
        auto _index = index;
        if (j < 0)
          _index[0] -= span.dimSize[0];
        if (j > 0)
          _index[0] += span.dimSize[0];
        if (k < 0)
          _index[1] -= span.dimSize[1];
        if (k > 0)
          _index[1] += span.dimSize[1];
        EXPECT_FALSE(span.inBounds(_index));
      }
  }

  std::vector<int64_t> visited(span.totalSize(), 0);
  for (auto index : span) {
    EXPECT_TRUE(span.inBounds(index));
    int64_t linear_index = span(index);
    EXPECT_EQ(visited[linear_index], 0);
    visited[linear_index] = 1;
  }
  EXPECT_TRUE(std::all_of(visited.begin(), visited.end(), [](int v) { return v == 1; }));
}

test(
    三维范围测试) {
  MDSpan<3> span = MDSpan<3>::create(2, 3, 4);
  EXPECT_EQ(span.totalSize(), 24);
  EXPECT_TRUE(span.inBounds(Vector<int64_t, 3>::create(0, 0, 0)));
  EXPECT_TRUE(span.inBounds(Vector<int64_t, 3>::create(1, 2, 3)));
  EXPECT_FALSE(span.inBounds(Vector<int64_t, 3>::create(-1, 0, 0)));
  EXPECT_FALSE(span.inBounds(Vector<int64_t, 3>::create(0, -1, 0)));
  EXPECT_FALSE(span.inBounds(Vector<int64_t, 3>::create(0, 0, -1)));
  EXPECT_FALSE(span.inBounds(Vector<int64_t, 3>::create(2, 0, 0)));
  EXPECT_FALSE(span.inBounds(Vector<int64_t, 3>::create(0, 3, 0)));
  EXPECT_FALSE(span.inBounds(Vector<int64_t, 3>::create(0, 0, 4)));
  EXPECT_EQ(span(Vector<int64_t, 3>::create(0, 0, 0)), 0);
  EXPECT_EQ(span(Vector<int64_t, 3>::create(1, 0, 0)), 1);
  EXPECT_EQ(span(Vector<int64_t, 3>::create(0, 1, 0)), 2);
  EXPECT_EQ(span(Vector<int64_t, 3>::create(0, 0, 1)), 6);
  EXPECT_EQ(span(Vector<int64_t, 3>::create(1, 2, 3)), 23);
  EXPECT_EQ((span[Vector<int64_t, 3>::create(0, 0, 0)]), 0);
  EXPECT_EQ((span[Vector<int64_t, 3>::create(1, 0, 0)]), 1);
  EXPECT_EQ((span[Vector<int64_t, 3>::create(0, 1, 0)]), 2);
  EXPECT_EQ((span[Vector<int64_t, 3>::create(0, 0, 1)]), 6);
  EXPECT_EQ((span[Vector<int64_t, 3>::create(1, 2, 3)]), 23);
  for (int64_t i = 0; i < span.totalSize(); i++) {
    Vector<int64_t, 3> index = span.toIndex(i);
    EXPECT_EQ(span(index), i);
    EXPECT_EQ((span[index]), i);
    EXPECT_EQ((index[0] + index[1] * span.dimSize[0] + index[2] * span.dimSize[0] * span.dimSize[1]), i);
    EXPECT_EQ((index[2]), i / (span.dimSize[0] * span.dimSize[1]));
    EXPECT_EQ((index[1]), (i / span.dimSize[0]) % span.dimSize[1]);
    EXPECT_EQ((index[0]), i % span.dimSize[0]);
    EXPECT_TRUE(span.inBounds(index));
    for (int j = -1; j <= 1; j++)
      for (int k = -1; k <= 1; k++)
        for (int l = -1; l <= 1; l++) {
          if (j == 0 && k == 0 && l == 0)
            continue;
          auto _index = index;
          if (j < 0)
            _index[0] -= span.dimSize[0];
          if (j > 0)
            _index[0] += span.dimSize[0];
          if (k < 0)
            _index[1] -= span.dimSize[1];
          if (k > 0)
            _index[1] += span.dimSize[1];
          if (l < 0)
            _index[2] -= span.dimSize[2];
          if (l > 0)
            _index[2] += span.dimSize[2];
          EXPECT_FALSE(span.inBounds(_index));
        }
  }
  std::vector<int64_t> visited(span.totalSize(), 0);
  for (auto index : span) {
    EXPECT_TRUE(span.inBounds(index));
    int64_t linear_index = span(index);
    EXPECT_EQ(visited[linear_index], 0);
    visited[linear_index] = 1;
  }
  EXPECT_TRUE(std::all_of(visited.begin(), visited.end(), [](int v) { return v == 1; }));
}
test(
    四维范围测试) {
  MDSpan<4> span = MDSpan<4>::create(2, 2, 3, 4);
  EXPECT_EQ(span.totalSize(), 48);
  EXPECT_TRUE(span.inBounds(Vector<int64_t, 4>::create(0, 0, 0, 0)));
  EXPECT_TRUE(span.inBounds(Vector<int64_t, 4>::create(1, 1, 2, 3)));
  EXPECT_FALSE(span.inBounds(Vector<int64_t, 4>::create(-1, 0, 0, 0)));
  EXPECT_FALSE(span.inBounds(Vector<int64_t, 4>::create(0, -1, 0, 0)));
  EXPECT_FALSE(span.inBounds(Vector<int64_t, 4>::create(0, 0, -1, 0)));
  EXPECT_FALSE(span.inBounds(Vector<int64_t, 4>::create(0, 0, 0, -1)));
  EXPECT_FALSE(span.inBounds(Vector<int64_t, 4>::create(2, 0, 0, 0)));
  EXPECT_FALSE(span.inBounds(Vector<int64_t, 4>::create(0, 2, 0, 0)));
  EXPECT_FALSE(span.inBounds(Vector<int64_t, 4>::create(0, 0, 3, 0)));
  EXPECT_FALSE(span.inBounds(Vector<int64_t, 4>::create(0, 0, 0, 4)));
  EXPECT_EQ(span(Vector<int64_t, 4>::create(0, 0, 0, 0)), 0);
  EXPECT_EQ(span(Vector<int64_t, 4>::create(1, 0, 0, 0)), 1);
  EXPECT_EQ(span(Vector<int64_t, 4>::create(0, 1, 0, 0)), 2);
  EXPECT_EQ(span(Vector<int64_t, 4>::create(0, 0, 1, 0)), 4);
  EXPECT_EQ(span(Vector<int64_t, 4>::create(0, 0, 0, 1)), 12);
  EXPECT_EQ(span(Vector<int64_t, 4>::create(1, 1, 2, 3)), 47);
  EXPECT_EQ((span[Vector<int64_t, 4>::create(0, 0, 0, 0)]), 0);
  EXPECT_EQ((span[Vector<int64_t, 4>::create(1, 0, 0, 0)]), 1);
  EXPECT_EQ((span[Vector<int64_t, 4>::create(0, 1, 0, 0)]), 2);
  EXPECT_EQ((span[Vector<int64_t, 4>::create(0, 0, 1, 0)]), 4);
  EXPECT_EQ((span[Vector<int64_t, 4>::create(0, 0, 0, 1)]), 12);
  EXPECT_EQ((span[Vector<int64_t, 4>::create(1, 1, 2, 3)]), 47);
  for (int64_t i = 0; i < span.totalSize(); i++) {
    Vector<int64_t, 4> index = span.toIndex(i);
    EXPECT_EQ(span(index), i);
    EXPECT_EQ((span[index]), i);
    EXPECT_EQ((index[0] + index[1] * span.dimSize[0] + index[2] * span.dimSize[0] * span.dimSize[1] +
               index[3] * span.dimSize[0] * span.dimSize[1] * span.dimSize[2]),
              i);
    EXPECT_EQ((index[3]), i / (span.dimSize[0] * span.dimSize[1] * span.dimSize[2]));
    EXPECT_EQ((index[2]), (i / (span.dimSize[0] * span.dimSize[1])) % span.dimSize[2]);
    EXPECT_EQ((index[1]), (i / span.dimSize[0]) % span.dimSize[1]);
    EXPECT_EQ((index[0]), i % span.dimSize[0]);
    EXPECT_TRUE(span.inBounds(index));
    for (int j = -1; j <= 1; j++)
      for (int k = -1; k <= 1; k++)
        for (int l = -1; l <= 1; l++)
          for (int m = -1; m <= 1; m++) {
            if (j == 0 && k == 0 && l == 0 && m == 0)
              continue;
            auto _index = index;
            if (j < 0)
              _index[0] -= span.dimSize[0];
            if (j > 0)
              _index[0] += span.dimSize[0];
            if (k < 0)
              _index[1] -= span.dimSize[1];
            if (k > 0)
              _index[1] += span.dimSize[1];
            if (l < 0)
              _index[2] -= span.dimSize[2];
            if (l > 0)
              _index[2] += span.dimSize[2];
            if (m < 0)
              _index[3] -= span.dimSize[3];
            if (m > 0)
              _index[3] += span.dimSize[3];
            EXPECT_FALSE(span.inBounds(_index));
          }
  }
  std::vector<int64_t> visited(span.totalSize(), 0);
  for (auto index : span) {
    EXPECT_TRUE(span.inBounds(index));
    int64_t linear_index = span(index);
    EXPECT_EQ(visited[linear_index], 0);
    visited[linear_index] = 1;
  }
  EXPECT_TRUE(std::all_of(visited.begin(), visited.end(), [](int v) { return v == 1; }));
}

test(
    二维起点非零范围测试) {
  MDBeginEndSpan<2> span =
      MDBeginEndSpan<2>::create(Vector<int64_t, 2>::create(1, 2), Vector<int64_t, 2>::create(4, 6));
  MDBeginEndSpan<2> span_ =
      MDBeginEndSpan<2>::create_from_center_size(Vector<int64_t, 2>::create(2, 4), Vector<int64_t, 2>::create(3, 4));
  EXPECT_TRUE(span.begins == span_.begins);
  EXPECT_TRUE(span.ends == span_.ends);
  EXPECT_EQ(span.totalSize(), 12);
  EXPECT_TRUE(span.inBounds(Vector<int64_t, 2>::create(1, 2)));
  EXPECT_TRUE(span.inBounds(Vector<int64_t, 2>::create(3, 5)));
  EXPECT_FALSE(span.inBounds(Vector<int64_t, 2>::create(0, 2)));
  EXPECT_FALSE(span.inBounds(Vector<int64_t, 2>::create(1, 1)));
  EXPECT_FALSE(span.inBounds(Vector<int64_t, 2>::create(4, 2)));
  EXPECT_FALSE(span.inBounds(Vector<int64_t, 2>::create(1, 6)));

  std::vector<int64_t> visited(span.totalSize(), 0);
  for (auto index : span) {
    EXPECT_TRUE(span.inBounds(index));
    int64_t linear_index = (index[0] - span.begins[0]) + (index[1] - span.begins[1]) * (span.ends[0] - span.begins[0]);
    EXPECT_EQ(visited[linear_index], 0);
    visited[linear_index] = 1;
  }
  EXPECT_TRUE(std::all_of(visited.begin(), visited.end(), [](int v) { return v == 1; }));
}

test(
    四维起点非零范围测试) {
  MDBeginEndSpan<4> span =
      MDBeginEndSpan<4>::create(Vector<int64_t, 4>::create(1, 2, 3, 4), Vector<int64_t, 4>::create(3, 4, 6, 8));
  MDBeginEndSpan<4> span_ = MDBeginEndSpan<4>::create_from_center_size(Vector<int64_t, 4>::create(2, 3, 4, 6),
                                                                       Vector<int64_t, 4>::create(2, 2, 3, 4));
  EXPECT_TRUE(span.begins == span_.begins);
  EXPECT_TRUE(span.ends == span_.ends);
  EXPECT_EQ(span.totalSize(), 48);
  EXPECT_TRUE(span.inBounds(Vector<int64_t, 4>::create(1, 2, 3, 4)));
  EXPECT_TRUE(span.inBounds(Vector<int64_t, 4>::create(2, 3, 5, 7)));
  EXPECT_TRUE(span.inBounds(Vector<int64_t, 4>::create(2, 3, 5, 7)));
  EXPECT_FALSE(span.inBounds(Vector<int64_t, 4>::create(0, 2, 3, 4)));
  EXPECT_FALSE(span.inBounds(Vector<int64_t, 4>::create(1, 1, 3, 4)));
  EXPECT_FALSE(span.inBounds(Vector<int64_t, 4>::create(1, 2, 2, 4)));
  EXPECT_FALSE(span.inBounds(Vector<int64_t, 4>::create(1, 2, 3, 3)));
  EXPECT_FALSE(span.inBounds(Vector<int64_t, 4>::create(3, 2, 3, 4)));
  EXPECT_FALSE(span.inBounds(Vector<int64_t, 4>::create(1, 4, 3, 4)));
  EXPECT_FALSE(span.inBounds(Vector<int64_t, 4>::create(1, 2, 6, 4)));
  EXPECT_FALSE(span.inBounds(Vector<int64_t, 4>::create(1, 2, 3, 8)));
  std::vector<int64_t> visited(span.totalSize(), 0);
  for (auto index : span) {
    EXPECT_TRUE(span.inBounds(index));
    int64_t linear_index =
        (index[0] - span.begins[0]) + (index[1] - span.begins[1]) * (span.ends[0] - span.begins[0]) +
        (index[2] - span.begins[2]) * (span.ends[0] - span.begins[0]) * (span.ends[1] - span.begins[1]) +
        (index[3] - span.begins[3]) * (span.ends[0] - span.begins[0]) * (span.ends[1] - span.begins[1]) *
            (span.ends[2] - span.begins[2]);
    EXPECT_EQ(visited[linear_index], 0);
    visited[linear_index] = 1;
  }
  EXPECT_TRUE(std::all_of(visited.begin(), visited.end(), [](int v) { return v == 1; }));
}