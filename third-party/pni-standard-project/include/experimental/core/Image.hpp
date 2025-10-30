#pragma once
#include "Span.hpp"
namespace openpni::experimental::core {
template <int D, FloatingPoint_c F = float>
struct Grids {
  static constexpr int dimension = D;
  using presicion = F;

  Vector<F, D> origin;  // 原点坐标
  Vector<F, D> spacing; // 体素大小（分辨率）
  MDSpan<D> size;       // 各维度的体素数量

  __PNI_CUDA_MACRO__ static Grids<D, F> create_by_origin_spacing_size(
      const Vector<F, D> &__origin, const Vector<F, D> &__spacing, const Vector<int64_t, D> &__voxel_nums) {
    Grids<D, F> result;
    result.origin = __origin;
    result.spacing = __spacing;
    result.size = MDSpan<D>::create(__voxel_nums);
    return result;
  }

  __PNI_CUDA_MACRO__ static Grids<D, F> create_by_center_spacing_size(
      const Vector<F, D> &__center, const Vector<F, D> &__spacing, const Vector<int64_t, D> &__voxel_nums) {
    Grids<D, F> result;
    result.spacing = __spacing;
    result.size = MDSpan<D>::create(__voxel_nums);
    result.origin = __center - __spacing * (result.size.dimSize.template to<F>() * F(0.5));
    return result;
  }

  __PNI_CUDA_MACRO__ static Grids<D, F> create_by_spacing_size(
      const Vector<F, D> &__spacing, const Vector<int64_t, D> &__voxel_nums) {
    return create_by_center_spacing_size(Vector<F, D>::create(F(0)), __spacing, __voxel_nums);
  }

  __PNI_CUDA_MACRO__ static Grids<D, F> create_by_center_boxLength_size(
      const Vector<F, D> &__center, const Vector<F, D> &__boundingBox, const Vector<int64_t, D> &__voxel_nums) {
    Grids<D, F> result;
    result.size = MDSpan<D>::create(__voxel_nums);
    result.spacing = __boundingBox / result.size.dimSize.template to<F>();
    result.origin = __center - result.spacing * (result.size.dimSize.template to<F>() * F(0.5));
    return result;
  }
  __PNI_CUDA_MACRO__ static Grids<D, F> create_by_origin_boxLength_size(
      const Vector<F, D> &__boundingBox, const Vector<int64_t, D> &__voxel_nums) {
    return create_by_center_boxLength_size(Vector<F, D>::create(F(0)), __boundingBox, __voxel_nums);
  }

  __PNI_CUDA_MACRO__ uint64_t totalSize() const {
    int64_t result = 1;
    for (int i = 0; i < D; i++)
      result *= size.dimSize[i];
    return static_cast<uint64_t>(result);
  }
  __PNI_CUDA_MACRO__ F volume() const {
    F result = 1.0f;
    for (int i = 0; i < D; i++)
      result *= spacing[i] * size.dimSize[i];
    return result;
  }
  __PNI_CUDA_MACRO__ Vector<Vector<F, D>, 2> bounding_box() const {
    return Vector<Vector<F, D>, 2>::create(origin, origin + spacing * size.dimSize.template to<F>());
  }
  __PNI_CUDA_MACRO__ Vector<F, D> boxLength() const { return spacing * size.dimSize.template to<F>(); }
  __PNI_CUDA_MACRO__ Vector<F, D> center() const { return origin + spacing * (size.dimSize.template to<F>() * F(0.5)); }
  __PNI_CUDA_MACRO__ Vector<F, D> end() const { return origin + spacing * size.dimSize.template to<F>(); }
  __PNI_CUDA_MACRO__ Grids<D, F> grid_point(
      Vector<int64_t, D> index) const {
    Grids<D, F> result;
    result.origin = origin + spacing * index.to(F());
    result.spacing = spacing;
    result.size = MDSpan<D>::create(1);
    return result;
  }
  template <FloatingPoint_c FF>
  __PNI_CUDA_MACRO__ Vector<int64_t, D> find_index_from_float(
      Vector<FF, D> const &point) const {
    Vector<int64_t, D> result;
    for (int i = 0; i < D; i++)
      result[i] = static_cast<int64_t>(core::FMath<FF>::ffloor((point[i] - origin[i]) / spacing[i]));
    return result;
  }
  __PNI_CUDA_MACRO__ bool in(
      Vector<int64_t, D> const &index) const {
    return size.inBounds(index);
  }
  template <FloatingPoint_c FF>
  __PNI_CUDA_MACRO__ bool in(
      Vector<FF, D> const &point) const {
    return in(find_index_from_float(point));
  }

  __PNI_CUDA_MACRO__ std::size_t at(
      const Vector<int64_t, D> &__index) const {
    std::size_t result = __index[D - 1];
    for (int i = D - 2; i >= 0; i--)
      result = result * size.dimSize[i] + __index[i];
    return result;
  }

  __PNI_CUDA_MACRO__ Vector<F, D> voxel_center(
      Vector<int64_t, D> const &index) const {
    return origin + spacing * (index.template to<F>() + Vector<F, D>::create(F(0.5)));
  }
  __PNI_CUDA_MACRO__ Vector<F, D> voxel_bounding_box(
      Vector<int64_t, D> const &index) const {
    return Vector<F, D>::create(origin + spacing * index.template to<F>(),
                                origin + spacing * (index.template to<F>() + Vector<F, D>::create(F(1.0))));
  }
  __PNI_CUDA_MACRO__ MDSpan<D> index_span() const { return MDSpan<D>::create(size.dimSize); }
  __PNI_CUDA_MACRO__ bool operator==(
      const Grids<D, F> &other) const {
    return origin == other.origin && spacing == other.spacing && size == other.size;
  }
};

template <typename T, int D>
struct TensorData {
  using ValueType = std::remove_const_t<T>;
  using GridType = Grids<D>;
  GridType grid;
  T *ptr;
};
template <typename T, int D>
using TensorDataInput = TensorData<const T, D>;
template <typename T, int D>
using TensorDataOutput = TensorData<std::remove_const_t<T>, D>;
template <typename T, int D>
struct TensorDataIO {
  using ValueType = std::remove_const_t<T>;
  using GridType = Grids<D>;
  using InputType = TensorData<const T, D>;
  using OutputType = TensorData<std::remove_const_t<T>, D>;
  GridType grid;
  T const *ptr_in;
  std::remove_const_t<T> *ptr_out;
  __PNI_CUDA_MACRO__ auto input_branch() const { return InputType{grid, ptr_in}; }
  __PNI_CUDA_MACRO__ auto output_branch() const { return OutputType{grid, ptr_out}; }
};

template <typename T>
using Image2D = TensorData<T, 2>;
template <typename T>
using Image2DInput = TensorDataInput<T, 2>;
template <typename T>
using Image2DOutput = TensorDataOutput<T, 2>;
template <typename T>
using Image2DIO = TensorDataIO<T, 2>;

template <typename T>
using Image3D = TensorData<T, 3>;
template <typename T>
using Image3DInput = TensorDataInput<T, 3>;
template <typename T>
using Image3DOutput = TensorDataOutput<T, 3>;
template <typename T>
using Image3DIO = TensorDataIO<T, 3>;

template <typename T>
using Image4D = TensorData<T, 4>;
template <typename T>
using Image4DInput = TensorDataInput<T, 4>;
template <typename T>
using Image4DOutput = TensorDataOutput<T, 4>;
template <typename T>
using Image4DIO = TensorDataIO<T, 4>;

} // namespace openpni::experimental::core
