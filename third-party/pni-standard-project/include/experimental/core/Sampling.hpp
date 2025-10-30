#pragma once
#include <chrono>
#include <limits>

#include "../../misc/Platform-Independent.hpp"
#include "BasicMath.hpp"
#include "Image.hpp"
#include "Mich.hpp"
namespace openpni::experimental::core {
template <FloatingPoint_c F>
struct SamplingUniform {
  __PNI_CUDA_MACRO__ static auto create(
      F from, F to, unsigned sample_num, F bias = F(0.5))
  // bias is supposed to be in [0,1), default is 0.5
  {
    SamplingUniform result;
    sample_num = FMath<int>::max(1, sample_num);
    result.m_current_index = 0;
    result.m_sample_num = sample_num;
    result.m_bias = bias;
    result.m_from = from;
    result.m_to = to;
    return result;
  }

  __PNI_CUDA_MACRO__ bool has_next() const noexcept { return m_current_index < m_sample_num; }
  __PNI_CUDA_MACRO__ Vector<F, 2> next() noexcept { // return (point, step_size / total_length)
    // It's caller's responsibility to ensure has_next() is true before calling next()
    F t = (F(m_current_index) + m_bias) / F(m_sample_num);
    F result = m_from * (F(1) - t) + m_to * t;
    m_current_index++;
    return Vector<F, 2>::create(result, (m_to - m_from) / F(m_sample_num));
  }

private:
  unsigned m_sample_num;
  unsigned m_current_index;
  F m_bias;
  F m_from;
  F m_to;
};

template <FloatingPoint_c F>
struct SamplingUniformWithTOF {
  __PNI_CUDA_MACRO__ static auto create(
      F from, F to, unsigned sample_num, F distance_cry1_cry2, Vector<int16_t, 2> pair_tofMean,
      Vector<int16_t, 2> pair_tofDev, int16_t deltaTOF, F bias = F(0.5)) {
    SamplingUniformWithTOF result;
    sample_num = FMath<int>::max(1, sample_num);
    result.m_current_index = 0;
    result.m_sample_num = sample_num;
    result.m_bias = bias;
    result.m_from = from;
    result.m_to = to;
    result.m_distance_cry1_cry2 = distance_cry1_cry2;

    result.m_tof_center =
        (deltaTOF + pair_tofMean[0] - pair_tofMean[1]) * misc::speed_of_light_ps / 2 + distance_cry1_cry2 / 2;

    result.m_gaussianStandardDeviation_mm = FMath<float>::fsqrt(FMath<float>::fpow(static_cast<F>(pair_tofDev[0]), 2) +
                                                                FMath<float>::fpow(static_cast<F>(pair_tofDev[1]), 2)) *
                                            misc::speed_of_light_ps;
    return result;
  }

  __PNI_CUDA_MACRO__ bool has_next() const noexcept { return m_current_index < m_sample_num; }
  __PNI_CUDA_MACRO__ Vector<F, 2> next() noexcept { // return (point, step_size / total_length)
    // It's caller's responsibility to ensure has_next() is true before calling next()
    F t = (F(m_current_index) + m_bias) / F(m_sample_num);
    F result = m_from * (F(1) - t) + m_to * t;

    // Calculate Gaussian integral for TOF weighting
    F guassianX1_alpha = m_from + (m_to - m_from) * F(m_current_index) / F(m_sample_num);
    F guassianX1_mm = guassianX1_alpha * m_distance_cry1_cry2;
    F guassianX2_alpha = m_from + (m_to - m_from) * F(m_current_index + 1) / F(m_sample_num);
    F guassianX2_mm = guassianX2_alpha * m_distance_cry1_cry2;
    F gaussianIntegral =
        FMath<F>::gauss_integral(guassianX1_mm, guassianX2_mm, m_tof_center, m_gaussianStandardDeviation_mm);

    m_current_index++;
    return Vector<F, 2>::create(result, gaussianIntegral);
  }

private:
  F m_from;
  F m_to;
  unsigned m_sample_num;
  unsigned m_current_index;
  F m_bias; // in [0,1)
  F m_tof_center;
  F m_gaussianStandardDeviation_mm;
  F m_distance_cry1_cry2; // cry1到cry2的距离
};

} // namespace openpni::experimental::core