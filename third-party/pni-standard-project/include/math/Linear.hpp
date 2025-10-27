#pragma once

namespace openpni::math {
struct _LinearFitting {
  template <typename CalculationPresicion>
  struct WithBias {
    using C = CalculationPresicion;
    int n = 0;
    C sumX = C(0);  // sum(x)
    C sumY = C(0);  // sum(y)
    C sumXY = C(0); // sum(x*y)
    C sumXX = C(0); // sum(x*x)

    __PNI_CUDA_MACRO__ C add(
        C __x, C __y) {
      sumX += __x;
      sumY += __y;
      sumXY += __x * __y;
      sumXX += __x * __x;
      n++;
    }
    __PNI_CUDA_MACRO__ C slope() { return (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX); }
    __PNI_CUDA_MACRO__ C intercept() { return (sumXX * sumY - sumX * sumXY) / (n * sumXX - sumX * sumX); }
    __PNI_CUDA_MACRO__ C predict(
        C __x) {
      return slope() * __x + intercept();
    }
  };
  template <typename CalculationPresicion>
  struct WithoutBias {
    using C = CalculationPresicion;
    C sumXY = C(0); // sum(x*y)
    C sumXX = C(0); // sum(x*x)

    __PNI_CUDA_MACRO__ C add(
        C __x, C __y) {
      sumXY += __x * __y;
      sumXX += __x * __x;
    }
    __PNI_CUDA_MACRO__ C slope() { return fabs(sumXY / sumXX); }
    __PNI_CUDA_MACRO__ C intercept() { return 0; }
    __PNI_CUDA_MACRO__ C predict(
        C __x) {
      return slope() * __x;
    }
  };
};
inline constexpr auto LinearFitting = _LinearFitting{};

} // namespace  openpni::math
