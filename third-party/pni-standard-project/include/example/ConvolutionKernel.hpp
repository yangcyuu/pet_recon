#pragma once
#include <cmath>
#include <numbers>

#include "../basic/Image.hpp"
#include "../basic/Math.hpp"
#include "../math/Convolution.hpp"
namespace openpni::example {
template <int U, int V, int W, typename Precision = double>
  requires std::is_floating_point_v<Precision>
process::ConvolutionKernelStatic<U, V, W, Precision> gaussianKernel(
    Precision __hwhm_x, Precision __hwhm_y, Precision __hwhm_z) // 半宽半高
{
  process::ConvolutionKernelStatic<U, V, W, Precision> result;

  constexpr auto halfU = U / 2;
  constexpr auto halfV = V / 2;
  constexpr auto halfW = W / 2;

  // 直接给出常数值，以下写法避免nvcc编译错误
  constexpr Precision sqrt2ln2 = 1.1774100225; // std::sqrt(2 * std::numbers::ln2_v<Precision>);
  constexpr Precision sqrt2pi = 2.5066282746;  // std::sqrt(2 * std::numbers::pi_v<Precision>);
  constexpr Precision sqrt2pi3 = sqrt2pi * sqrt2pi * sqrt2pi;

  const Precision sigmaX = __hwhm_x / sqrt2ln2;
  const Precision sigmaY = __hwhm_y / sqrt2ln2;
  const Precision sigmaZ = __hwhm_z / sqrt2ln2;

  Precision sum = 0;
  for (int k = -result.sizeW() / 2; k <= result.sizeW() / 2; k++)
    for (int j = -result.sizeV() / 2; j <= result.sizeV() / 2; j++)
      for (int i = -result.sizeU() / 2; i <= result.sizeU() / 2; i++) {
        Precision coff = i * i / (sigmaX * sigmaX) + j * j / (sigmaY * sigmaY) + k * k / (sigmaZ * sigmaZ);
        coff = exp(-.5 * coff) / (sqrt2pi3 * sigmaX * sigmaY * sigmaZ);
        result.at(i + halfU, j + halfV, k + halfW) = coff;
        sum += coff;
      }
  for (int k = -result.sizeW() / 2; k <= result.sizeW() / 2; k++)
    for (int j = -result.sizeV() / 2; j <= result.sizeV() / 2; j++)
      for (int i = -result.sizeU() / 2; i <= result.sizeU() / 2; i++)
        result.at(i + halfU, j + halfV, k + halfW) /= sum;

  return result;
}

template <int U, int V, int W, typename Precision = double>
  requires std::is_floating_point_v<Precision>
process::ConvolutionKernelStatic<U, V, W, Precision> gaussianKernel(
    Precision __hfwhm) {
  return gaussianKernel<U, V, W, Precision>(__hfwhm, __hfwhm, __hfwhm);
}

template <int Size, typename Precision = double>
process::ConvolutionKernelStatic<Size, Size, Size, Precision> gaussianKernel(
    Precision __hfwhm) {
  return gaussianKernel<Size, Size, Size, Precision>(__hfwhm);
}

template <int Size, typename Precision = double>
process::ConvolutionKernelStatic<Size, Size, Size, Precision> gaussianKernel(
    Precision __hwhm_x, Precision __hwhm_y, Precision __hwhm_z) {
  return gaussianKernel<Size, Size, Size, Precision>(__hwhm_x, __hwhm_y, __hwhm_z);
}

template <int Size, typename Precision = double>
process::ConvolutionKernelStatic<Size, Size, Size, Precision> gaussianKernel(
    basic::Vec3<Precision> __hwhm) {
  return gaussianKernel<Size, Size, Size, Precision>(__hwhm.x, __hwhm.y, __hwhm.z);
}

// this kernel is from OSEM_V3
template <int U, int V, int W, typename Precision = double>
  requires std::is_floating_point_v<Precision>
process::ConvolutionKernelStatic<U, V, W, Precision> GaussionC3Dkernel(
    Precision __hwhm,
    const openpni::basic::Vec3<float> voxelSize) // 半宽半高
{
  process::ConvolutionKernelStatic<U, V, W, Precision> result;

  constexpr auto halfU = U / 2;
  constexpr auto halfV = V / 2;
  constexpr auto halfW = W / 2;

  const Precision sigma = __hwhm / 2.355f;
  float pi = 3.1415926535f;

  Precision sum = 0;
  for (int k = -halfW; k <= halfW; k++)
    for (int j = -halfV; j <= halfV; j++)
      for (int i = -halfU; i <= halfU; i++) {
        Precision coff = i * i * (voxelSize.x * voxelSize.x) + j * j * (voxelSize.y * voxelSize.y) +
                         k * k * (voxelSize.z * voxelSize.z);
        coff = coff / (2 * sigma * sigma);
        coff = exp(-1 * coff) / (2 * pi * sigma * sigma);
        result.kernel[i + halfU][j + halfV][k + halfW] = coff;
        sum += coff;
      }
  for (int k = -halfW; k <= halfW; k++)
    for (int j = -halfV; j <= halfV; j++)
      for (int i = -halfU; i <= halfU; i++)
        result.kernel[i + halfU][j + halfV][k + halfW] /= sum;

  return result;
}
template <int Size, typename Precision = double>
process::ConvolutionKernelStatic<Size, Size, Size, Precision> GaussionC3Dkernel(
    Precision __hfwhm, const openpni::basic::Vec3<float> voxelSize) {
  return GaussionC3Dkernel<Size, Size, Size, Precision>(__hfwhm, voxelSize);
}
} // namespace openpni::example
