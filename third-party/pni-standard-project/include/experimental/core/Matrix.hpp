#pragma once
#include "Vector.hpp"
namespace openpni::experimental::core {
template <typename T, int Rows, int Cols>
struct Matrix {
  T data[Rows][Cols];

  __PNI_CUDA_MACRO__ static Matrix<T, Rows, Cols> create() {
    Matrix<T, Rows, Cols> result;
    return result;
  }

  __PNI_CUDA_MACRO__ static Matrix<T, Rows, 1> from_column_vector(
      const Vector<T, Rows> &v) {
    Matrix<T, Rows, 1> result;
    for (int r = 0; r < Rows; r++)
      result.data[r][0] = v[r];
    return result;
  }

  __PNI_CUDA_MACRO__ static Matrix<T, 1, Cols> from_row_vector(
      const Vector<T, Cols> &v) {
    Matrix<T, 1, Cols> result;
    for (int c = 0; c < Cols; c++)
      result.data[0][c] = v[c];
    return result;
  }

  __PNI_CUDA_MACRO__ T &operator()(
      int64_t r, int64_t c) {
    return data[r][c];
  }
  __PNI_CUDA_MACRO__ const T &operator()(
      int64_t r, int64_t c) const {
    return data[r][c];
  }

  __PNI_CUDA_MACRO__ Vector<T, Rows> take_column(
      int64_t c) const {
    Vector<T, Rows> result;
    for (int64_t r = 0; r < Rows; r++)
      result[r] = data[r][c];
    return result;
  }

  __PNI_CUDA_MACRO__ Vector<T, Cols> take_row(
      int64_t r) const {
    Vector<T, Cols> result;
    for (int64_t c = 0; c < Cols; c++)
      result[c] = data[r][c];
    return result;
  }

  template <typename TT = T>
  __PNI_CUDA_MACRO__ auto operator*(
      const Vector<TT, Cols> &other) const -> Vector<decltype(TT() * T()), Rows> {
    Vector<decltype(TT() * T()), Rows> result;
    for (int r = 0; r < Rows; r++) {
      result[r] = decltype(TT() * T())(0);
      for (int c = 0; c < Cols; c++)
        result[r] = result[r] + data[r][c] * other[c];
    }
    return result;
  }

  template <typename TT = T, int OtherCols>
  __PNI_CUDA_MACRO__ auto operator*(
      const Matrix<TT, Cols, OtherCols> &other) const -> Matrix<decltype(TT() * T()), Rows, OtherCols> {
    Matrix<decltype(TT() * T()), Rows, OtherCols> result;
    for (int r = 0; r < Rows; r++) {
      for (int c = 0; c < OtherCols; c++) {
        result(r, c) = decltype(TT() * T())(0);
        for (int k = 0; k < Cols; k++)
          result(r, c) = result(r, c) + data[r][k] * other(k, c);
      }
    }
    return result;
  }
};

} // namespace openpni::experimental::core