#pragma once
#include <execution>

#include "../../basic/CpuInfo.hpp"
#include "../core/BasicMath.hpp"
#include "../core/Image.hpp"
#include "../tools/Parallel.hpp"
namespace openpni::experimental::algorithms {
template <FloatingPoint_c F, int D>
class GaussianConvolutionCPU {
private:
  static std::vector<F> create_guassian_array(
      F grid_spacing, int max_array_length, F hwhm) noexcept {
    std::vector<F> result;
    const F sqrt2ln2 = core::FMath<F>::fsqrt(2.0 * core::FMath<F>::flog(2.0));
    const F max_sigma = F(5.0);
    const F sigma = hwhm / sqrt2ln2 / grid_spacing;
    const int max_sigma_array_length = odd_or_1(core::FMath<F>::fceil(max_sigma * 2.0 * sigma));
    max_array_length = std::min(max_array_length, max_sigma_array_length);
    result.resize(max_array_length);
    result[max_array_length / 2] = core::FMath<F>::gauss_integral(F(-0.5), F(0.5), 0, sigma);
    for (int i = 1; i < max_array_length / 2; i++) {
      const F val = core::FMath<F>::gauss_integral(F(i - 0.5), F(i + 0.5), 0, sigma);
      result[max_array_length / 2 + i] = val;
      result[max_array_length / 2 - i] = val;
    }
    return result;
  }

public:
  using HWHMType = core::Vector<F, D>; // Half Width at Half Maximum
public:
  GaussianConvolutionCPU(
      core::Vector<F, D> const &__hwhm) noexcept
      : m_hwhm(__hwhm) {}
  GaussianConvolutionCPU(
      F __hwhm) noexcept
      : m_hwhm(core::Vector<F, D>::create(__hwhm)) {}
  GaussianConvolutionCPU() noexcept
      : m_hwhm(core::Vector<F, D>::create(F(1))) {}
  GaussianConvolutionCPU<F, D> copy() const noexcept {
    GaussianConvolutionCPU c;
    c.m_hwhm = m_hwhm;
    return c;
  }

public:
  void setHWHM(
      core::Vector<F, D> const &__hwhm) noexcept {
    m_hwhm = __hwhm;
  }
  void setHWHM(
      F __hwhm) noexcept {
    m_hwhm = core::Vector<F, D>::create(__hwhm);
  }

private:
  struct EmbeddedArray {
    std::vector<F> data;
    core::Grids<D> grid;
  };
  template <int d>
  EmbeddedArray create_guassian_kernel(
      std::vector<F> &&__guassian_array) const noexcept {
    EmbeddedArray result;
    result.data = std::move(__guassian_array);
    result.grid.origin = core::Vector<F, D>::create();  // not used
    result.grid.spacing = core::Vector<F, D>::create(); // not used
    result.grid.size = core::MDSpan<D>::create();
    for (int i = 0; i < D; i++)
      if (i != d)
        result.grid.size.dimSize[i] = 1;
      else
        result.grid.size.dimSize[d] = static_cast<int64_t>(result.data.size());
    return result;
  }
  static inline int odd_or_1(
      int v) noexcept {
    return v % 2 == 0 ? v + 1 : v;
  }

  template <int d>
  void conv_1d(
      core::Grids<D, F> __grids, const F *__in, F *__out) const noexcept {
    const EmbeddedArray embedded_kernel = create_guassian_kernel<d>(
        create_guassian_array(__grids.spacing[d], odd_or_1(__grids.size.dimSize[d]), m_hwhm[d]));
    if (embedded_kernel.data.size() <= 1) { // No convolution needed
      std::copy(__in, __in + __grids.totalSize(), __out);
      return;
    }
    tools::parallel_for_each(__grids.index_span(), [&](core::Vector<int64_t, D> const &index) noexcept {
      F sum_coef = 0;
      F sum_value = 0;
      const auto delta_span = core::MDBeginEndSpan<D>::create_from_center_size(core::Vector<int64_t, D>::create(),
                                                                               embedded_kernel.grid.size.dimSize);
      for (const auto index_delta : delta_span) {
        auto index_image = index + index_delta;
        if (!__grids.size.inBounds(index_image))
          continue;
        const auto coef = embedded_kernel.data[embedded_kernel.grid.size(index_delta - delta_span.begins)];
        sum_coef += coef;
        sum_value += __in[__grids.size[index_image]] * coef;
      }
      __out[__grids.size[index]] = sum_value / sum_coef;
    });
  }
  template <int d>
    requires(d >= 0 && d < D)
  void conv_impl(
      core::Grids<D, F> __grids, const F *__in, F *__temp1, F *__temp2, F *__out) const noexcept {
    if constexpr (D == 1) { // 处理特殊情况，当输入输出指针相等时需要使用缓冲区
      if (__in != __out)
        conv_1d<d>(__grids, __in, __out);
      else {
        conv_1d<d>(__grids, __in, __temp1);
        std::copy(__temp1, __temp1 + __grids.totalSize(), __out);
      }
      return;
    } else {
      if constexpr (d == 0) {
        conv_1d<d>(__grids, __in, __temp1); // The first dimension, input to __temp1
        conv_impl<d + 1>(__grids, __in, __temp1, __temp2, __out);
        return;
      } else if constexpr (d == D - 1) {
        conv_1d<d>(__grids, __temp1, __out); // The last dimension, output to __out
        return;
      } else {
        conv_1d<d>(__grids, __temp1, __temp2); // Middle dimensions, ping-pong between __temp1 and __temp2
        conv_impl<d + 1>(__grids, __in, __temp2, __temp1, __out); // Next dimension
        return;
      }
    }
  }

public:
  void conv(
      core::TensorDataIO<F, D> __image) const noexcept {
    if (__image.grid.totalSize() > m_bufferSize) {
      m_bufferSize = __image.grid.totalSize();
      m_buffer = std::make_unique_for_overwrite<F[]>(__image.grid.totalSize());
      m_pairBuffer = std::make_unique_for_overwrite<F[]>(__image.grid.totalSize());
    }
    conv_impl<0>(__image.grid, __image.ptr_in, m_buffer.get(), m_pairBuffer.get(), __image.ptr_out);
  };
  void deconv(
      core::TensorDataIO<F, D> __image) const noexcept {
    conv(__image); // Gaussian convolution is symmetric, so convolution and deconvolution are the same
  };

private:
  core::Vector<F, D> m_hwhm; // 半高半宽
  mutable std::unique_ptr<F[]> m_buffer;
  mutable std::unique_ptr<F[]> m_pairBuffer;
  mutable std::size_t m_bufferSize{0};
};
} // namespace openpni::experimental::algorithms
