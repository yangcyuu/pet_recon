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
      F from, F to, unsigned sample_num, F distance_p1_p2, F bias = F(0.5))
  // bias is supposed to be in [0,1), default is 0.5
  {
    SamplingUniform result;
    sample_num = FMath<int>::max(1, sample_num);
    result.m_current_index = 0;
    result.m_sample_num = sample_num;
    result.m_bias = bias;
    result.m_from = from;
    result.m_to = to;
    result.m_distance_p1_p2 = distance_p1_p2;
    return result;
  }

  __PNI_CUDA_MACRO__ bool has_next() const noexcept { return m_current_index < m_sample_num; }
  __PNI_CUDA_MACRO__ Vector<F, 2> next() noexcept { // return (point, step_size / total_length)
    // It's caller's responsibility to ensure has_next() is true before calling next()
    F t = (F(m_current_index) + m_bias) / F(m_sample_num);
    F result = m_from * (F(1) - t) + m_to * t;
    m_current_index++;
    return Vector<F, 2>::create(result, (m_to - m_from) / F(m_sample_num) * m_distance_p1_p2);
  }

private:
  unsigned m_sample_num;
  unsigned m_current_index;
  F m_bias;
  F m_from;
  F m_to;

  F m_distance_p1_p2; // 图像交点的距离
};

template <FloatingPoint_c F>
struct SamplingUniformWithTOF {
  __PNI_CUDA_MACRO__ static auto create(
      F from, F to, unsigned sample_num, F distance_cry1_cry2, Vector<int16_t, 2> pair_tofMean,
      Vector<int16_t, 2> pair_tofDev, int16_t deltaTOF, int16_t TOFBinWid_ps, F bias = F(0.5)) {
    SamplingUniformWithTOF result;
    sample_num = FMath<int>::max(1, sample_num);
    result.m_current_index = 0;
    result.m_sample_num = sample_num;
    result.m_bias = bias;
    result.m_from = from;
    result.m_to = to;
    result.m_distance_cry1_cry2 = distance_cry1_cry2;

    result.m_gaussianStandardDeviation_mm = FMath<float>::fsqrt(FMath<float>::fpow(static_cast<F>(pair_tofDev[0]), 2) +
                                                                FMath<float>::fpow(static_cast<F>(pair_tofDev[1]), 2)) *
                                            misc::speed_of_light_ps;
    float TOFBinWidth_mm = TOFBinWid_ps * misc::speed_of_light_ps / 2.0;
    float alpha_TOFbinCenter =
        floor((deltaTOF + TOFBinWid_ps * 0.5) / TOFBinWid_ps) * TOFBinWidth_mm / distance_cry1_cry2 + 0.5;
    result.alpha_TOFbinLeft = alpha_TOFbinCenter - TOFBinWidth_mm * 0.5 / distance_cry1_cry2;
    result.alpha_TOFbinRight = alpha_TOFbinCenter + TOFBinWidth_mm * 0.5 / distance_cry1_cry2;
    result.alpha_min = alpha_TOFbinCenter - 3 * result.m_gaussianStandardDeviation_mm / distance_cry1_cry2;
    result.alpha_max = alpha_TOFbinCenter + 3 * result.m_gaussianStandardDeviation_mm / distance_cry1_cry2;

    result.m_tof_center =
        (deltaTOF + pair_tofMean[0] - pair_tofMean[1]) * misc::speed_of_light_ps / 2 + distance_cry1_cry2 / 2;

    return result;
  }

  __PNI_CUDA_MACRO__ bool has_next() const noexcept { return m_current_index < m_sample_num; }
  __PNI_CUDA_MACRO__ Vector<F, 2> next() noexcept { // return (point, step_size / total_length)
    // It's caller's responsibility to ensure has_next() is true before calling next()
    F t = (F(m_current_index) + m_bias) / F(m_sample_num);
    F result = m_from * (F(1) - t) + m_to * t;

    // Calculate TOF weight using Siddon-like sampling with error function
    // ac_before: current voxel left border (in alpha coordinate)
    // ac: current voxel right border (in alpha coordinate)
    F lengtha = (m_to - m_from) / F(m_sample_num);
    F ac_before = m_from + (m_to - m_from) * F(m_current_index) / F(m_sample_num);
    F ac = ac_before + lengtha;

    if (ac < alpha_min || ac_before > alpha_max) {
      m_current_index++;
      return Vector<F, 2>::create(result, F(0));
    }

    // Calculate distances to TOF bin boundaries
    // F alpha_mid = ac_before + F(0.5) * lengtha;
    F dis2TOFbinLeft = (alpha_TOFbinLeft - result) * m_distance_cry1_cry2;
    F dis2TOFbinRight = (alpha_TOFbinRight - result) * m_distance_cry1_cry2;

    // TOF weight using error function: 0.5 * (erf(right/sigma/sqrt(2)) - erf(left/sigma/sqrt(2)))
    constexpr F gauss_sigma_inv_sqrt2 = F(1.0) / F(1.41421356237);
    F tofWeight =
        F(0.5) * (FMath<float>::ferf(dis2TOFbinRight / m_gaussianStandardDeviation_mm * gauss_sigma_inv_sqrt2) -
                  FMath<float>::ferf(dis2TOFbinLeft / m_gaussianStandardDeviation_mm * gauss_sigma_inv_sqrt2));

    m_current_index++;
    return Vector<F, 2>::create(result, lengtha * tofWeight * m_distance_cry1_cry2);
  }

  __PNI_CUDA_MACRO__ F get_alpha_min() const noexcept { return alpha_min; }
  __PNI_CUDA_MACRO__ F get_alpha_max() const noexcept { return alpha_max; }

private:
  F m_from;
  F m_to;
  unsigned m_sample_num;
  unsigned m_current_index;
  F m_bias; // in [0,1)
  F m_tof_center;
  F m_gaussianStandardDeviation_mm;
  F m_distance_cry1_cry2; // cry1到cry2的距离

  F alpha_TOFbinLeft;
  F alpha_TOFbinRight;

  F alpha_min;
  F alpha_max;
};

} // namespace openpni::experimental::core