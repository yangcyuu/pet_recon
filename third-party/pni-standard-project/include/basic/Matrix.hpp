#pragma once
#include <cmath>

#include "Math.hpp"
#include "Point.hpp"
namespace openpni::basic {
template <typename T>
struct Matrix2D {
  Vec2<T> a0; // [a0.x  a1.x]
  Vec2<T> a1; // [a0.y  a1.y]

  template <typename TT>
  __PNI_CUDA_MACRO__ auto operator*(
      const Vec2<T> &b) // 矩阵直接左乘向量
      -> Vec2<decltype(a0.x * b.x)> const {
    return Vec2<decltype(a0.x * b.x)>{a0.x * b.x + a1.x * b.y, a0.y * b.x + a1.y * b.y};
  }

  template <typename TT = T> // 向量先转置，然后被矩阵右乘
  __PNI_CUDA_MACRO__ friend auto operator*(
      const Vec2<TT> &b, const Matrix2D<TT> &m) -> Vec2<decltype(b.x * m.a0.x)> {
    return Vec2<decltype(b.x * m.a0.x)>{b.x * m.a0.x + b.y * m.a0.y, b.x * m.a1.x + b.y * m.a1.y};
  }

  __PNI_CUDA_MACRO__ Matrix2D<T> Transpose() const // 转置
  {
    Matrix2D<T> result;
    result.a0.x = a0.x;
    result.a0.y = a1.x;
    result.a1.x = a0.y;
    result.a1.y = a1.y;
    return result;
  }
};

template <typename T>
struct Matrix3D {
  Vec3<T> a0; // [a0.x  a1.x  a2.x]
  Vec3<T> a1; // [a0.y  a1.y  a2.y]
  Vec3<T> a2; // [a0.z  a1.z  a2.z]

  template <typename TT>
  __PNI_CUDA_MACRO__ auto operator*(
      const Vec3<T> &b) // 矩阵直接左乘向量
      -> Vec3<decltype(a0.x * b.x)> const {
    return Vec3<decltype(a0.x * b.x)>{
        a0.x * b.x + a1.x * b.y + a2.x * b.z, // 计算 x 分量
        a0.y * b.x + a1.y * b.y + a2.y * b.z, // 计算 y 分量
        a0.z * b.x + a1.z * b.y + a2.z * b.z  // 计算 z 分量
    };
  }

  template <typename TT = T> // 向量先转置，然后被矩阵右乘
  __PNI_CUDA_MACRO__ friend auto operator*(
      const Vec3<TT> &b, const Matrix3D<TT> &m) -> Vec3<decltype(b.x * m.a0.x)> {
    return Vec3<decltype(b.x * m.a0.x)>{
        b.x * m.a0.x + b.y * m.a0.y + b.z * m.a0.z, // 计算 x 分量
        b.x * m.a1.x + b.y * m.a1.y + b.z * m.a1.z, // 计算 y 分量
        b.x * m.a2.x + b.y * m.a2.y + b.z * m.a2.z  // 计算 z 分量
    };
  }

  __PNI_CUDA_MACRO__ Matrix3D<T> Transpose() const // 转置
  {
    Matrix3D<T> result;
    result.a0.x = a0.x;
    result.a0.y = a1.x;
    result.a0.z = a2.x;

    result.a1.x = a0.y;
    result.a1.y = a1.y;
    result.a1.z = a2.y;

    result.a2.x = a0.z;
    result.a2.y = a1.z;
    result.a2.z = a2.z;

    return result;
  }
};

template <FloatingPoint_c T>
__PNI_CUDA_MACRO__ Matrix2D<T> rotationMatrix(
    T __angle) // 如果需要应用旋转矩阵，则应该 Vector * Matrix
{
  T cosTheta = std::cos(__angle);
  T sinTheta = std::sin(__angle);

  // 构造旋转矩阵
  return Matrix2D<T>{
      {cosTheta, -sinTheta}, // 第一列
      {sinTheta, cosTheta}   // 第二列
  };
}

template <FloatingPoint_c T>
__PNI_CUDA_MACRO__ Matrix3D<T> rotationMatrixZ(
    T angle) // 绕 Z 轴的旋转矩阵
{
  T cosTheta = std::cos(angle);
  T sinTheta = std::sin(angle);

  // 构造旋转矩阵
  return Matrix3D<T>{
      {cosTheta, -sinTheta, 0}, // 第一列
      {sinTheta, cosTheta, 0},  // 第二列
      {0, 0, 1}                 // 第三列
  };
}
} // namespace openpni::basic
