#pragma once
#include "../basic/Image.hpp"
#include "../basic/Math.hpp"
#include "../basic/Matrix.hpp"
#include "../basic/Point.hpp"
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <functional>
#include <iostream>
#include <thread>
namespace openpni::process {
namespace _impl {
template <typename T>
concept IsFloatingPoint2D = FloatingPoint2D_c<T>;
template <typename T>
concept IsFloatingPoint3D = FloatingPoint2D_c<T>;

template <typename T>
concept IsTransformAction2DFloat = requires(T t, basic::Vec2<float> p) {
  { t | p } -> IsFloatingPoint2D;
  { p | t } -> IsFloatingPoint2D;
  { -t } -> std::same_as<T>;
};

template <typename T>
concept IsTransformAction2DDouble = requires(T t, basic::Vec2<float> p) {
  { t | p } -> IsFloatingPoint2D;
  { p | t } -> IsFloatingPoint2D;
  { -t } -> std::same_as<T>;
};
template <typename T>
concept ContractTransformAction2D =
    IsTransformAction2DDouble<T> || IsTransformAction2DFloat<T>;

template <typename T>
concept IsTransformAction3DFloat = requires(T t, basic::Vec3<float> p) {
  { t | p } -> IsFloatingPoint3D;
  { p | t } -> IsFloatingPoint3D;
  { -t } -> std::same_as<T>;
};
template <typename T>
concept IsTransformAction3DDouble = requires(T t, basic::Vec3<float> p) {
  { t | p } -> IsFloatingPoint3D;
  { p | t } -> IsFloatingPoint3D;
  { -t } -> std::same_as<T>;
};
template <typename T>
concept IsTransformAction3D = IsTransformAction3DDouble<T> || IsTransformAction3DFloat<T>;

template <typename T>
concept IsTransformAction = IsTransformAction2DDouble<T> || IsTransformAction3DFloat<T>;

template <typename T>
concept IsPoint = requires { typename T::value_type; } &&
                  (std::same_as<T, basic::Vec2<typename T::value_type>> ||
                   std::same_as<T, basic::Vec3<typename T::value_type>>);

template <typename T>
        concept IsInterpolationMethod2D =
            requires { typename T::value_type; } &&
            (requires(const T &_, const basic::Vec2<float> &p, const typename T::value_type *img, const basic::Image2DGeometry &g) {
            { _(p, img, g) } -> std::same_as<typename T::value_type>; } || requires(const T &_, const basic::Vec2<double> &p, const typename T::value_type *img, const basic::Image2DGeometry &g) {
            { _(p, img, g) } -> std::same_as<typename T::value_type>; });

template <bool ForwardDirection = true, IsPoint Point, typename... Actions>
  requires(IsTransformAction<Actions> && ...)
__PNI_CUDA_MACRO__ Point applyTransforms_impl(const Point &point,
                                              const Actions &...actions) {
  if constexpr (ForwardDirection)
    return basic::make_vec2<typename Point::value_type>((point | ... | actions));
  return basic::make_vec2<typename Point::value_type>((actions | ... | point));
}

template <typename InputValueType, typename OutputValueType,
          IsInterpolationMethod2D InterpolationMethod, typename... Actions>
  requires(IsTransformAction<Actions> && ...)
__PNI_CUDA_MACRO__ inline void
transform2D_impl(const InputValueType *__input, OutputValueType *__output,
                 const basic::Image2DGeometry &__ig, const basic::Image2DGeometry &__og,
                 const InterpolationMethod &__ipMethod, int __x, int __y,
                 const Actions &...negativedActions) {
  const auto pointOfOutput =
      basic::make_vec2<typename InterpolationMethod::value_type>(__x + .5, __y + .5) *
          __og.voxelSize +
      __og.imgBegin;
  const auto pointOfInput =
      applyTransforms_impl<false>(pointOfOutput, negativedActions...);
  __output[__og.at(__x, __y)] = __ipMethod(pointOfInput, __input, __ig);
}

template <typename InputValueType, typename OutputValueType,
          IsInterpolationMethod2D InterpolationMethod, typename... Actions>
  requires(IsTransformAction<Actions> && ...)
__global__ void
transform2D_kernel(const InputValueType *__input, OutputValueType *__output,
                   basic::Image2DGeometry __ig, basic::Image2DGeometry __og,
                   InterpolationMethod __ipMethod, Actions... negativedActions) {
  unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= __og.voxelNum.x || y >= __og.voxelNum.y)
    return;

  const auto pointOfOutput =
      basic::make_vec2<typename InterpolationMethod::value_type>(x + .5, y + .5) *
          __og.voxelSize +
      __og.imgBegin;
  const auto pointOfInput =
      applyTransforms_impl<false>(pointOfOutput, negativedActions...);
  __output[__og.at(x, y)] = __ipMethod(pointOfInput, __input, __ig);
}

} // namespace _impl

template <typename InputValueType, typename OutputValueType,
          _impl::IsInterpolationMethod2D InterpolationMethod, typename... Actions>
  requires(_impl::IsTransformAction<Actions> && ...)
void transform2D(const InputValueType *__input, OutputValueType *__output,
                 const basic::Image2DGeometry &__ig, const basic::Image2DGeometry &__og,
                 const InterpolationMethod &__ipMethod, std::size_t maxThreadNum,
                 const Actions &...actions) {
#pragma omp parallel for num_threads(std::max<int>(1, maxThreadNum))
  for (int __x = 0; __x < __og.voxelNum.x; __x++)
    for (int __y = 0; __y < __og.voxelNum.y; __y++)
      _impl::transform2D_impl(__input, __output, __ig, __og, __ipMethod, __x, __y,
                              (-actions)...);
}

template <FloatingPoint_c T>
struct ActionTranslate2D {
  typedef T value_type;
  basic::Vec2<T> a;
  template <FloatingPoint_c TT>
  __PNI_CUDA_MACRO__ auto operator|(const basic::Vec2<TT> &b) const -> decltype(a + b) {
    return a + b;
  }
  template <FloatingPoint_c TT>
  __PNI_CUDA_MACRO__ friend auto operator|(const basic::Vec2<TT> &point,
                                           const ActionTranslate2D<T> &action) {
    return action | point;
  }
  ActionTranslate2D<T> operator-() const { return ActionTranslate2D{-a}; }
};
template <FloatingPoint_c T>
struct ActionRotate2D {
  typedef T value_type;
  basic::Vec2<T> center;
  basic::Matrix2D<T> rotationMatrix;
  template <FloatingPoint_c TT>
  __PNI_CUDA_MACRO__ auto operator|(basic::Vec2<TT> a) const {
    return (a - center) * rotationMatrix + center;
  }
  template <FloatingPoint_c TT>
  __PNI_CUDA_MACRO__ friend auto operator|(basic::Vec2<TT> point,
                                           const ActionRotate2D<T> &action) {
    return action | point;
  }
  __PNI_CUDA_MACRO__ ActionRotate2D<T> operator-() const {
    return ActionRotate2D{center, rotationMatrix.Transpose()};
  }
};
template <FloatingPoint_c T>
struct ActionScale2D {
  typedef T value_type;
  basic::Vec2<T> center;
  basic::Vec2<T> scale;
  template <FloatingPoint_c TT>
  __PNI_CUDA_MACRO__ auto operator|(const basic::Vec2<TT> &a) const {
    return (a - center) * scale + center;
  }
  template <FloatingPoint_c TT>
  __PNI_CUDA_MACRO__ friend auto operator|(basic::Vec2<TT> point,
                                           const ActionScale2D<T> &action) {
    return action | point;
  }
  __PNI_CUDA_MACRO__ ActionRotate2D<T> operator-() const {
    return ActionRotate2D{center, 1 / scale};
  }
};
static_assert(std::is_trivially_copyable_v<ActionTranslate2D<float>>,
              "struct ActionTranslate2D must be trivially copyable, so that can be used "
              "in cuda kernels.");
static_assert(std::is_trivially_copyable_v<ActionRotate2D<float>>,
              "struct ActionRotate2D must be trivially copyable, so that can be used in "
              "cuda kernels.");
static_assert(std::is_trivially_copyable_v<ActionScale2D<float>>,
              "struct ActionScale2D must be trivially copyable, so that can be used in "
              "cuda kernels.");
} // namespace openpni::process
