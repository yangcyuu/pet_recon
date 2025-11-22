#pragma once
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <functional>
#include <iostream>
#include <thread>

#include "../basic/Image.hpp"
#include "../basic/Math.hpp"
#include "../basic/Matrix.hpp"
#include "../basic/Point.hpp"
namespace openpni::process {
struct ActionTranslate2D {
  typedef float value_type;
  basic::Vec2<float> a;
  __PNI_CUDA_MACRO__ auto operator|(
      const basic::Vec2<float> &b) const -> decltype(a + b) {
    return a + b;
  }
  __PNI_CUDA_MACRO__ friend auto operator|(
      const basic::Vec2<float> &point, const ActionTranslate2D &action) {
    return action | point;
  }
  ActionTranslate2D operator-() const { return ActionTranslate2D{-a}; }
};
struct ActionRotate2D {
  typedef float value_type;
  basic::Vec2<float> center;
  basic::Matrix2D<float> rotationMatrix;
  __PNI_CUDA_MACRO__ auto operator|(
      basic::Vec2<float> a) const {
    return (a - center) * rotationMatrix + center;
  }
  __PNI_CUDA_MACRO__ friend auto operator|(
      basic::Vec2<float> point, const ActionRotate2D &action) {
    return action | point;
  }
  __PNI_CUDA_MACRO__ ActionRotate2D operator-() const { return ActionRotate2D{center, rotationMatrix.Transpose()}; }
};
struct ActionScale2D {
  typedef float value_type;
  basic::Vec2<float> center;
  basic::Vec2<float> scale;
  __PNI_CUDA_MACRO__ auto operator|(
      const basic::Vec2<float> &a) const {
    return (a - center) * scale + center;
  }
  __PNI_CUDA_MACRO__ friend auto operator|(
      basic::Vec2<float> point, const ActionScale2D &action) {
    return action | point;
  }
  __PNI_CUDA_MACRO__ ActionScale2D operator-() const {
    return ActionScale2D{center, basic::make_vec2<float>(1.0f / scale.x, 1.0f / scale.y)};
  }
};
} // namespace openpni::process
