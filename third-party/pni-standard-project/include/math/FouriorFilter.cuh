#pragma once
#include <cufft.h>

// #include "../Define.h"
#include "../basic/CudaPtr.hpp"
#include "../basic/Math.hpp"
#include "../basic/Point.hpp"
#ifndef PI
#define PI 3.14159265358979323846
#endif
namespace openpni::process {
template <FloatingPoint_c Precision>
struct CuFFTPrecisionAdapter {};
template <>
struct CuFFTPrecisionAdapter<float> {
  typedef cufftComplex cufft_type;
  typedef cufftReal cufftReal;
};
template <>
struct CuFFTPrecisionAdapter<double> {
  typedef cufftDoubleComplex cufft_type;
  typedef cufftReal cufftDoubleReal;
};

template <typename CuFFTValueType, typename InputValueType>
__global__ void kernel_real2Complex(
    InputValueType const *__inputPtr, CuFFTValueType *__outputPtr, uint64_t __num) {
  uint64_t tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid >= __num)
    return;
  __outputPtr[tid].x = __inputPtr[tid];
  __outputPtr[tid].y = 0.0;
}

template <typename CuFFTValueType, typename OutputValueType>
__global__ void kernel_complex2Real(
    const CuFFTValueType *__input, OutputValueType *__output, uint64_t __num) {
  uint64_t tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid >= __num)
    return;

  __output[tid] = __input[tid].x;
}

template <FloatingPoint_c Precision, typename FouriorFilterMethod>
void fouriorFilter(
    Precision const *__d_inputPtr, Precision *__d_outputPtr, basic::Vec2<unsigned> __size,
    FouriorFilterMethod &__filter) {
  const auto datasetSize = uint64_t(__size.x) * __size.y;
  auto d_complex = make_cuda_sync_ptr<typename CuFFTPrecisionAdapter<Precision>::cufft_type>(datasetSize);
  auto d_complex_buffer = make_cuda_sync_ptr<typename CuFFTPrecisionAdapter<Precision>::cufft_type>(datasetSize);

  kernel_real2Complex<typename CuFFTPrecisionAdapter<Precision>::cufft_type, Precision>
      <<<(datasetSize + 255) / 256, 256>>>(__d_inputPtr, d_complex.get(), datasetSize);

  cufftHandle mFFTHandle;
  int n[1] = {static_cast<int>(__size.x)};
  int inembed[2] = {static_cast<int>(__size.x), static_cast<int>(__size.y)};
  int onembed[2] = {static_cast<int>(__size.x), static_cast<int>(__size.y)};
  cufftPlanMany(&mFFTHandle, 1, n, inembed, 1, static_cast<int>(__size.x), onembed, 1, static_cast<int>(__size.x),
                CUFFT_C2C,
                static_cast<int>(__size.y) // batch
  );
  cufftExecC2C(mFFTHandle, d_complex.get(), d_complex_buffer.get(), CUFFT_FORWARD);
  std::swap(d_complex, d_complex_buffer);
  __filter(d_complex.get(), d_complex_buffer.get());
  std::swap(d_complex, d_complex_buffer);
  cufftExecC2C(mFFTHandle, d_complex.get(), d_complex_buffer.get(), CUFFT_INVERSE);
  std::swap(d_complex, d_complex_buffer);
  cufftDestroy(mFFTHandle);
  kernel_complex2Real<typename CuFFTPrecisionAdapter<Precision>::cufft_type, Precision>
      <<<(datasetSize + 255) / 256, 256>>>(d_complex.get(), __d_outputPtr, datasetSize);
  kernel_complex2Real<typename CuFFTPrecisionAdapter<Precision>::cufft_type, Precision>
      <<<(datasetSize + 255) / 256, 256>>>(d_complex.get(), __d_outputPtr, datasetSize);
}

inline __global__ void kernel_applyFilter(
    const cufftComplex *__d_filter, const cufftComplex *__d_input, cufftComplex *__d_output, const unsigned __uSize,
    const unsigned __vSize, const unsigned __cutoffLength) {
  const auto tid = uint64_t(threadIdx.x) + blockIdx.x * blockDim.x;
  const auto totalNum = uint64_t(__uSize) * __vSize;
  if (tid >= totalNum)
    return;

  float real = 0.0;
  int u = tid % __uSize;

  if (__cutoffLength <= u && u < __uSize - __cutoffLength)
    real = 0.0;
  else
    real = __d_filter[u].x * 2;

  __d_output[tid].x = __d_input[tid].x * real;
  __d_output[tid].y = __d_input[tid].y * real;
}

template <FloatingPoint_c Precision>
struct FouriorCutoffFilter {
  typedef typename CuFFTPrecisionAdapter<Precision>::cufft_type CuFFTType;

  const CuFFTType *d_filter;
  unsigned uSize;
  unsigned vSize;
  unsigned cutoffLength;

  void operator()(
      const CuFFTType *__d_input, CuFFTType *__d_output) {
    const auto totalNum = uint64_t(uSize) * vSize;
    kernel_applyFilter<<<(totalNum + 255) / 256, 256>>>(d_filter, __d_input, __d_output, uSize, vSize, cutoffLength);
  }
};

template <FloatingPoint_c Precision>
bool initializeFouriorFilter(
    typename CuFFTPrecisionAdapter<Precision>::cufft_type *__d_filter, const unsigned __sizeU) {
  using CuFFTType = typename CuFFTPrecisionAdapter<Precision>::cufft_type;
  std::vector<float> f(__sizeU, 0.25f);
  std::vector<int> n(__sizeU / 2);
  for (int i = 0; i < __sizeU / 2 / 2; i++) {
    n[i] = 2 * i + 1;
    n[__sizeU / 2 - 1 - i] = 2 * i + 1;
  }
  for (int i = 1; i < __sizeU; i++)
    f[i] = (i % 2 == 1) ? (-1.0 / ((PI * n[i / 2]) * (PI * n[i / 2]))) : 0.0;

  std::vector<CuFFTType> h_filter_host(__sizeU);
  for (unsigned u = 0; u < __sizeU; ++u) {
    h_filter_host[u].x = f[u] / 4000.0f; // 实部
    h_filter_host[u].y = 0.0f;           // 虚部
  }
  cudaMemcpy(__d_filter, h_filter_host.data(), __sizeU * sizeof(CuFFTType), cudaMemcpyHostToDevice);
  cufftHandle fftHandle;
  cufftPlan1d(&fftHandle, __sizeU, CUFFT_C2C, 1);
  cufftExecC2C(fftHandle, __d_filter, __d_filter, CUFFT_FORWARD);
  cufftDestroy(fftHandle);
  return true;
}
} // namespace openpni::process

// #include "../Undef.h"
