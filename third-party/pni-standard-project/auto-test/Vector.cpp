#include "include/experimental/core/Vector.hpp"

#include <gtest/gtest.h>
#define TEST_SUITE_NAME openpni_experimental_core_Vector
#define test(name) TEST(TEST_SUITE_NAME, name)
using namespace openpni::experimental::core;
using namespace openpni::experimental::algorithms;

test(
    标量测试1) {
  Vector<int, 1> v = Vector<int, 1>::create(5);
  EXPECT_EQ(v[0], 5);
  Vector<float, 1> vf = v.to<float>();
  EXPECT_FLOAT_EQ(vf[0], 5.0f);
  Vector<double, 1> vd = v.to<double>();
  EXPECT_DOUBLE_EQ(vd[0], 5.0);
  Vector<int, 1> vi = vd.to<int>();
  EXPECT_EQ(vi[0], 5);
  Vector<int, 1> v2 = apply(v, [](int x) { return x * 2; });
  EXPECT_EQ(v2[0], 10);
}

test(
    标量测试2) {
  Vector<float, 1> v = Vector<float, 1>::create(3.5f);
  EXPECT_FLOAT_EQ(v[0], 3.5f);
  Vector<int, 1> vi = v.to<int>();
  EXPECT_EQ(vi[0], 3); // 测试向下取整
  Vector<double, 1> vd = v.to<double>();
  EXPECT_DOUBLE_EQ(vd[0], 3.5);
  Vector<float, 1> vf = vd.to<float>();
  EXPECT_FLOAT_EQ(vf[0], 3.5f);
  Vector<float, 1> v2 = apply(v, [](float x) { return x + 1.5f; });
  EXPECT_FLOAT_EQ(v2[0], 5.0f);
}

test(
    二维向量测试) {
  Vector<int, 2> v = Vector<int, 2>::create(3, 4);
  EXPECT_EQ(v[0], 3);
  EXPECT_EQ(v[1], 4);
  Vector<float, 2> vf = v.to<float>();
  EXPECT_FLOAT_EQ(vf[0], 3.0f);
  EXPECT_FLOAT_EQ(vf[1], 4.0f);
  Vector<double, 2> vd = v.to<double>();
  EXPECT_DOUBLE_EQ(vd[0], 3.0);
  EXPECT_DOUBLE_EQ(vd[1], 4.0);
  Vector<int, 2> vi = vd.to<int>();
  EXPECT_EQ(vi[0], 3);
  EXPECT_EQ(vi[1], 4);
  Vector<int, 2> v2 = apply(v, [](int x) { return x * x; });
  EXPECT_EQ(v2[0], 9);
  EXPECT_EQ(v2[1], 16);
}

test(
    三维向量测试) {
  Vector<int, 3> v = Vector<int, 3>::create(1, 2, 3);
  EXPECT_EQ(v[0], 1);
  EXPECT_EQ(v[1], 2);
  EXPECT_EQ(v[2], 3);
  Vector<float, 3> vf = v.to<float>();
  EXPECT_FLOAT_EQ(vf[0], 1.0f);
  EXPECT_FLOAT_EQ(vf[1], 2.0f);
  EXPECT_FLOAT_EQ(vf[2], 3.0f);
  Vector<double, 3> vd = v.to<double>();
  EXPECT_DOUBLE_EQ(vd[0], 1.0);
  EXPECT_DOUBLE_EQ(vd[1], 2.0);
  EXPECT_DOUBLE_EQ(vd[2], 3.0);
  Vector<int, 3> vi = vd.to<int>();
  EXPECT_EQ(vi[0], 1);
  EXPECT_EQ(vi[1], 2);
  EXPECT_EQ(vi[2], 3);
  Vector<int, 3> v2 = apply(v, [](int x) { return x + 1; });
  EXPECT_EQ(v2[0], 2);
  EXPECT_EQ(v2[1], 3);
  EXPECT_EQ(v2[2], 4);
}

test(
    向量扩展与缩小测试) {
  Vector<int, 2> v2 = Vector<int, 2>::create(1, 2);
  Vector<int, 4> v4 = v2.right_expand(3, 4);
  EXPECT_EQ(v4[0], 1);
  EXPECT_EQ(v4[1], 2);
  EXPECT_EQ(v4[2], 3);
  EXPECT_EQ(v4[3], 4);
  Vector<int, 3> v3 = v4.right_shrink<1>();
  EXPECT_EQ(v3[0], 1);
  EXPECT_EQ(v3[1], 2);
  EXPECT_EQ(v3[2], 3);
  Vector<int, 2> v2b = v3.right_shrink<1>();
  EXPECT_EQ(v2b[0], 1);
  EXPECT_EQ(v2b[1], 2);
}

test(
    加法与减法测试) {
  Vector<int, 3> v1 = Vector<int, 3>::create(1, 2, 3);
  Vector<int, 3> v2 = Vector<int, 3>::create(4, 5, 6);
  Vector<int, 3> v3 = v1 + v2;
  EXPECT_EQ(v3[0], 5);
  EXPECT_EQ(v3[1], 7);
  EXPECT_EQ(v3[2], 9);
  Vector<int, 3> v4 = v1 - v2;
  EXPECT_EQ(v4[0], -3);
  EXPECT_EQ(v4[1], -3);
  EXPECT_EQ(v4[2], -3);
  Vector<int, 3> v5 = v1 + 10;
  EXPECT_EQ(v5[0], 11);
  EXPECT_EQ(v5[1], 12);
  EXPECT_EQ(v5[2], 13);
  Vector<int, 3> v6 = 20 + v1;
  EXPECT_EQ(v6[0], 21);
  EXPECT_EQ(v6[1], 22);
  EXPECT_EQ(v6[2], 23);
  Vector<int, 3> v7 = v2 - 2;
  EXPECT_EQ(v7[0], 2);
  EXPECT_EQ(v7[1], 3);
  EXPECT_EQ(v7[2], 4);
}
test(
    乘法与除法测试) {
  Vector<int, 3> v1 = Vector<int, 3>::create(1, 2, 3);
  Vector<int, 3> v2 = Vector<int, 3>::create(4, 5, 6);
  Vector<int, 3> v3 = v1 * v2;
  EXPECT_EQ(v3[0], 4);
  EXPECT_EQ(v3[1], 10);
  EXPECT_EQ(v3[2], 18);
  Vector<int, 3> v4 = v2 / v1;
  EXPECT_EQ(v4[0], 4);
  EXPECT_EQ(v4[1], 2);
  EXPECT_EQ(v4[2], 2);
  Vector<int, 3> v5 = v1 * 3;
  EXPECT_EQ(v5[0], 3);
  EXPECT_EQ(v5[1], 6);
  EXPECT_EQ(v5[2], 9);
  Vector<int, 3> v6 = 4 * v1;
  EXPECT_EQ(v6[0], 4);
  EXPECT_EQ(v6[1], 8);
  EXPECT_EQ(v6[2], 12);
  Vector<int, 3> v7 = v2 / 2;
  EXPECT_EQ(v7[0], 2);
  EXPECT_EQ(v7[1], 2);
  EXPECT_EQ(v7[2], 3);
}
test(
    一元加减测试) {
  Vector<int, 3> v1 = Vector<int, 3>::create(1, -2, 3);
  Vector<int, 3> v2 = +v1;
  EXPECT_EQ(v2[0], 1);
  EXPECT_EQ(v2[1], -2);
  EXPECT_EQ(v2[2], 3);
  Vector<int, 3> v3 = -v1;
  EXPECT_EQ(v3[0], -1);
  EXPECT_EQ(v3[1], 2);
  EXPECT_EQ(v3[2], -3);
}

test(
    点积与范数测试) {
  Vector<int, 4> v1 = Vector<int, 4>::create(1, 2, 3, 4);
  Vector<int, 4> v2 = Vector<int, 4>::create(5, 6, 7, 8);
  int dot = v1.dot(v2);
  EXPECT_EQ(dot, 70); // 1*5 + 2*6 + 3*7 + 4*8 = 70
  int l22 = v1.l22();
  EXPECT_EQ(l22, 30); // 1^2 + 2^2 + 3^2 + 4^2 = 30
  int l1 = v1.l1();
  EXPECT_EQ(l1, 10); // 1 + 2 + 3 + 4 = 10
}

test(
    其他函数测试) {
  Vector<double, 3> v1 = Vector<double, 3>::create(1.5, 2.5, 3.5);
  EXPECT_DOUBLE_EQ(v1.min(), 1.5);
  EXPECT_DOUBLE_EQ(v1.max(), 3.5);
  EXPECT_DOUBLE_EQ(v1.l1(), 7.5);
  EXPECT_DOUBLE_EQ(v1.l22(), 20.75);
  EXPECT_DOUBLE_EQ(l2(v1), std::sqrt(20.75));
  Vector<double, 3> v2 = normalized(v1);
  double len = l2(v2);
  EXPECT_NEAR(len, 1.0, 1e-9);
  Vector<double, 3> v3 = v1.reverse();
  EXPECT_DOUBLE_EQ(v3[0], 3.5);
  EXPECT_DOUBLE_EQ(v3[1], 2.5);
  EXPECT_DOUBLE_EQ(v3[2], 1.5);
}

test(
    三维叉乘测试) {
  Vector<double, 3> ex = Vector<double, 3>::create(1.0, 0.0, 0.0);
  Vector<double, 3> ey = Vector<double, 3>::create(0.0, 1.0, 0.0);
  Vector<double, 3> ez = Vector<double, 3>::create(0.0, 0.0, 1.0);
  Vector<double, 3> e1 = cross(ex, ey);
  EXPECT_DOUBLE_EQ(e1[0], 0.0);
  EXPECT_DOUBLE_EQ(e1[1], 0.0);
  EXPECT_DOUBLE_EQ(e1[2], 1.0);
  Vector<double, 3> e2 = cross(ey, ez);
  EXPECT_DOUBLE_EQ(e2[0], 1.0);
  EXPECT_DOUBLE_EQ(e2[1], 0.0);
  EXPECT_DOUBLE_EQ(e2[2], 0.0);
  Vector<double, 3> e3 = cross(ez, ex);
  EXPECT_DOUBLE_EQ(e3[0], 0.0);
  EXPECT_DOUBLE_EQ(e3[1], 1.0);
  EXPECT_DOUBLE_EQ(e3[2], 0.0);
  Vector<double, 3> exex = cross(ex, ex);
  EXPECT_DOUBLE_EQ(exex[0], 0.0);
  EXPECT_DOUBLE_EQ(exex[1], 0.0);
  EXPECT_DOUBLE_EQ(exex[2], 0.0);
  Vector<double, 3> eyey = cross(ey, ey);
  EXPECT_DOUBLE_EQ(eyey[0], 0.0);
  EXPECT_DOUBLE_EQ(eyey[1], 0.0);
  EXPECT_DOUBLE_EQ(eyey[2], 0.0);
  Vector<double, 3> ezez = cross(ez, ez);
  EXPECT_DOUBLE_EQ(ezez[0], 0.0);
  EXPECT_DOUBLE_EQ(ezez[1], 0.0);
  EXPECT_DOUBLE_EQ(ezez[2], 0.0);
  Vector<double, 3> pi_6 = Vector<double, 3>::create(std::cos(M_PI / 6), std::sin(M_PI / 6), 0.0);
  Vector<double, 3> pi_3 = Vector<double, 3>::create(std::cos(M_PI / 3), std::sin(M_PI / 3), 0.0);
  Vector<double, 3> pi_2 = Vector<double, 3>::create(std::cos(M_PI / 2), std::sin(M_PI / 2), 0.0);
  Vector<double, 3> pi = Vector<double, 3>::create(std::cos(M_PI), std::sin(M_PI), 0.0);
  Vector<double, 3> pi_6_pi_3 = cross(pi_6, pi_3);
  EXPECT_NEAR(pi_6_pi_3[0], 0.0, 1e-9);
  EXPECT_NEAR(pi_6_pi_3[1], 0.0, 1e-9);
  EXPECT_NEAR(pi_6_pi_3[2], std::sin(M_PI / 6), 1e-9);
  Vector<double, 3> pi_3_pi_2 = cross(pi_3, pi_2);
  EXPECT_NEAR(pi_3_pi_2[0], 0.0, 1e-9);
  EXPECT_NEAR(pi_3_pi_2[1], 0.0, 1e-9);
  EXPECT_NEAR(pi_3_pi_2[2], std::sin(M_PI / 6), 1e-9);
  Vector<double, 3> pi_2_pi = cross(pi_2, pi);
  EXPECT_NEAR(pi_2_pi[0], 0.0, 1e-9);
  EXPECT_NEAR(pi_2_pi[1], 0.0, 1e-9);
  EXPECT_NEAR(pi_2_pi[2], std::sin(M_PI / 2), 1e-9);
}