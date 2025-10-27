#pragma once
#include "GaussianConvolutionCPU.hpp"
#include "include/basic/CudaPtr.hpp"
#include "include/experimental/tools/Parallel.cuh"
namespace openpni::experimental::algorithms::impl {
template <FloatingPoint_c F, int D>
inline void conv_1d_impl(
    core::Grids<D, F> __gridsEmbedded, F const *__valueEmbedded, core::Grids<D, F> __grids, const F *__in, F *__out) {
  tools::parallel_for_each_CUDA(__grids.index_span(), [=] __device__(core::Vector<int64_t, D> const &index) {
    F sum_coef = 0;
    F sum_value = 0;
    const auto delta_span = core::MDBeginEndSpan<D>::create_from_center_size(core::Vector<int64_t, D>::create(),
                                                                             __gridsEmbedded.size.dimSize);
    for (const auto index_delta : delta_span) {
      auto index_image = index + index_delta;
      if (!__grids.size.inBounds(index_image))
        continue;
      const auto coef = __valueEmbedded[__gridsEmbedded.size(index_delta - delta_span.begins)];
      sum_coef += coef;
      sum_value += __in[__grids.size[index_image]] * coef;
    }
    __out[__grids.size[index]] = sum_value / sum_coef;
  });
}
} // namespace openpni::experimental::algorithms::impl
namespace openpni::experimental::algorithms {
namespace impl {
inline int odd_or_1(
    int v) noexcept {
  return v % 2 == 0 ? v + 1 : v;
}
template <FloatingPoint_c F>
inline int fill_guassian_array(
    cuda_sync_ptr<F> &out, F grid_spacing, int max_array_length, F hwhm) noexcept {
  const F sqrt2ln2 = core::FMath<F>::fsqrt(2.0 * core::FMath<F>::flog(2.0));
  const F max_sigma = F(5.0);
  const F sigma = hwhm / sqrt2ln2 / grid_spacing;
  const int max_sigma_array_length = odd_or_1(core::FMath<F>::fceil(max_sigma * 2.0 * sigma));
  max_array_length = std::min(odd_or_1(max_array_length), max_sigma_array_length);
  if (out.elements() < max_array_length)
    out.reserve(max_array_length);
  tools::parallel_for_each_CUDA(max_array_length, [=, out = out.data()] __device__(int i) {
    if (i == max_array_length / 2) {
      out[i] = core::FMath<F>::gauss_integral(F(-0.5), F(0.5), 0, sigma);
    } else {
      const int dist = abs(i - max_array_length / 2);
      out[i] = core::FMath<F>::gauss_integral(F(dist - 0.5), F(dist + 0.5), 0, sigma);
    }
  });
  return max_array_length;
}
} // namespace impl
template <FloatingPoint_c F, int D>
class GaussianConvolutionCUDA {
public:
  using HWHMType = core::Vector<F, D>; // Half Width at Half Maximum
public:
  GaussianConvolutionCUDA(
      core::Vector<F, D> const &__hwhm) noexcept
      : m_hwhm(__hwhm) {}
  GaussianConvolutionCUDA(
      F __hwhm) noexcept
      : m_hwhm(core::Vector<F, D>::create(__hwhm)) {}
  GaussianConvolutionCUDA() noexcept
      : m_hwhm(core::Vector<F, D>::create(F(1))) {}
  GaussianConvolutionCUDA copy() {
    GaussianConvolutionCUDA c;
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
    cuda_sync_ptr<F> *data;
    core::Grids<D> grid;
  };
  template <int d>
  EmbeddedArray create_guassian_kernel(
      int size) const noexcept {
    EmbeddedArray result;
    result.grid.origin = core::Vector<F, D>::create();  // not used
    result.grid.spacing = core::Vector<F, D>::create(); // not used
    result.grid.size = core::MDSpan<D>::create();
    for (int i = 0; i < D; i++)
      if (i != d)
        result.grid.size.dimSize[i] = 1;
      else
        result.grid.size.dimSize[d] = static_cast<int64_t>(size);
    result.data = &m_GaussBuffer;
    return result;
  }

  template <int d>
  void conv_1d(
      core::Grids<D, F> __grids, const F *__in, F *__out) const noexcept {
    const EmbeddedArray embedded_kernel = create_guassian_kernel<d>(
        impl::fill_guassian_array(m_GaussBuffer, __grids.spacing[d], __grids.size.dimSize[d], m_hwhm[d]));
    if (embedded_kernel.grid.totalSize() <= 1) { // No convolution needed
      cudaMemcpyAsync(__out, __in, sizeof(F) * __grids.totalSize(), cudaMemcpyDeviceToDevice);
      return;
    }
    impl::conv_1d_impl<F, D>(embedded_kernel.grid, embedded_kernel.data->data(), __grids, __in, __out);
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
        cudaMemcpyAsync(__out, __temp1, sizeof(F) * __grids.totalSize(), cudaMemcpyDeviceToDevice);
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
      m_buffer.reserve(__image.grid.totalSize());
      m_pairBuffer.reserve(__image.grid.totalSize());
    }
    conv_impl<0>(__image.grid, __image.ptr_in, m_buffer.get(), m_pairBuffer.get(), __image.ptr_out);
  };
  void deconv(
      core::TensorDataIO<F, D> __image) const noexcept {
    conv(__image); // Gaussian convolution is symmetric, so convolution and deconvolution are the same
  };

private:
  core::Vector<F, D> m_hwhm; // 半高半宽
  mutable cuda_sync_ptr<F> m_buffer{"GaussianConvolution_TempBuffer"};
  mutable cuda_sync_ptr<F> m_pairBuffer{"GaussianConvolution_TempBuffer2"};
  mutable cuda_sync_ptr<F> m_GaussBuffer{"GaussianConvolution_GaussBuffer"};
  mutable std::size_t m_bufferSize{0};
};
} // namespace openpni::experimental::algorithms