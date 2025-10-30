#pragma once
#include "../basic/DataView.hpp"
#include "../basic/Image.hpp"
#include "../basic/Math.hpp"
#include "../basic/Point.hpp"
#include "../math/TimeOfFlight.hpp"
#include "../misc/Platform-Independent.hpp"
namespace openpni::math {
// template <typename T>
// concept IsSamplingMethod = requires(T &sampler) {
//   {
//     sampler.next()
//   } -> FloatingPoint_c下一个采样值, 这个采样值是线段参数方程的参数值，范围
//                                     // 是 [0, 1]，0表示开始点，1表示结束点
//                                     {sampler.lastStepSize()}
//                                         ->FloatingPoint_c次采样的步长{sampler.isEnd()}
//                                         ->std::same_as<bool>; // 采样结束了
// };

template <FloatingPoint_c T>
struct SamplingIntersection {
  __PNI_CUDA_MACRO__ SamplingIntersection()
      : a(T(0))
      , lastStepSize(T(0)) {}

  __PNI_CUDA_MACRO__ int test(
      const basic::Event<T> &__projectionEvent, const basic::Image3DGeometry *__image3dSize,
      const basic::Vec2<p3df> *__roi = nullptr) {
    using basic::sgn;
    using basic::smallFloat;
    const p3df cryPoint1 = __projectionEvent.crystal1.geometry->position;
    const p3df cryPoint2 = __projectionEvent.crystal2.geometry->position;
    const auto cry2_cry1 = cryPoint2 - cryPoint1;

    const auto roi = __roi ? basic::cube_and(*__roi, basic::cube_absolute(__image3dSize->roi()))
                           : basic::cube_absolute(__image3dSize->roi());
    const auto [amin, amax] = basic::liang_barskey(&roi, &cryPoint1, &cryPoint2);
    if (amin >= amax)
      return 0;
    return 1;
  }

  __PNI_CUDA_MACRO__ bool setInfo(
      const basic::Event<T> &__projectionEvent, const basic::Image3DGeometry *__image3dSize,
      const cubef *__roi = nullptr) {
    using basic::sgn;
    using basic::smallFloat;
    const p3df cryPoint1 = __projectionEvent.crystal1.geometry->position;
    const p3df cryPoint2 = __projectionEvent.crystal2.geometry->position;
    const auto cry2_cry1 = cryPoint2 - cryPoint1;

    roi = __roi ? basic::cube_and(*__roi, basic::cube_absolute(__image3dSize->roi()))
                : basic::cube_absolute(__image3dSize->roi());
    const auto [amin, amax] = basic::liang_barskey(&roi, &cryPoint1, &cryPoint2);
    if (amax <= amin)
      return false;

    const float distance_cry1_cry2 = cry2_cry1.l2();
    distance_p1_p2 = distance_cry1_cry2 * (amax - amin);

    // 缓存常用值
    p1 = cryPoint1 + cry2_cry1 * amin;
    p2 = cryPoint1 + cry2_cry1 * amax;
    p2_p1 = p2 - p1;
    voxelSize = __image3dSize->voxelSize;
    imgBegin = __image3dSize->imgBegin;

    // 预计算倒数，避免运行时除法
    inv_p2_p1.x = 1 / (smallFloat(p2_p1.x));
    inv_p2_p1.y = 1 / (smallFloat(p2_p1.y));
    inv_p2_p1.z = 1 / (smallFloat(p2_p1.z));

    // 预计算方向符号
    direction_sign.x = sgn(p2_p1.x >= 0);
    direction_sign.y = sgn(p2_p1.y >= 0);
    direction_sign.z = sgn(p2_p1.z >= 0);

    // 预计算最小步长
    minStep = 0.1 / __image3dSize->voxelNum.l1();

    a = T(0);
    lastStepSize = T(0);

    return true;
  }
  __PNI_CUDA_MACRO__ void reset() {
    a = T(0);
    lastStepSize = T(0);
  }
  __PNI_CUDA_MACRO__ p3df next() {
    const auto pointNow = p1 + p2_p1 * a;

    const auto indexNow = (pointNow - imgBegin) / voxelSize;
    const auto indexTest = indexNow + p3df(direction_sign.x, direction_sign.y, direction_sign.z);
    basic::Vec3<T> a3Next = (voxelSize.pointWiseMul(indexTest) + imgBegin - p1) * inv_p2_p1;
    const auto aNext = fminf(fmaxf(fminf(fminf(a3Next.x, a3Next.y), a3Next.z), a + minStep), 1);
    // const auto aNext = fmaxf(fminf(fminf(a3Next.x, a3Next.y), a3Next.z), a + minStep);
    //  const auto aNext = a + minStep; // 这里简化处理，直接返回结束点
    lastStepSize = aNext - a;
    const auto nextPoint = p1 + p2_p1 * (a + aNext) * T(0.5);

    a = aNext;
    if (!roiIn(nextPoint))
      lastStepSize = 0;

    return nextPoint;
  }
  __PNI_CUDA_MACRO__ T getlastStepSize() const { return lastStepSize * distance_p1_p2; }
  __PNI_CUDA_MACRO__ bool isEnd() const { return a >= T(1); }
  __PNI_CUDA_MACRO__ bool roiIn(
      const p3df &__p) const {
    return roi.x.x <= __p.x && __p.x < roi.y.x && roi.x.y <= __p.y && __p.y <= roi.y.y && roi.x.z <= __p.z &&
           __p.z <= roi.y.z;
  }

private:
  p3df p1, p2;
  p3df p2_p1;     // 方向向量
  p3df inv_p2_p1; // 预计算的倒数，避免除法
  p3df voxelSize;
  p3df imgBegin;
  basic::Vec3<int> direction_sign; // 预计算的符号
  cubef roi;                       // 预计算的ROI
  cubef m_sample_roi;              // 采样对应片段
  float minStep;                   // 预计算的最小步长

  T a;              // 线段参数方程的参数值，范围是 [0, 1]
  T lastStepSize;   // 上一次采样的步长
  T distance_p1_p2; // p1到p2的距离
};

template <FloatingPoint_c T>
struct SamplingUniform {
  __PNI_CUDA_MACRO__ int test(
      const basic::Event<T> &__projectionEvent, const basic::Image3DGeometry *__image3dSize,
      const basic::Vec2<p3df> *__roi = nullptr) {
    using basic::sgn;
    using basic::smallFloat;
    const p3df cryPoint1 = __projectionEvent.crystal1.geometry->position;
    const p3df cryPoint2 = __projectionEvent.crystal2.geometry->position;
    const auto cry2_cry1 = cryPoint2 - cryPoint1;

    const auto roi = __roi ? basic::cube_and(*__roi, basic::cube_absolute(__image3dSize->roi()))
                           : basic::cube_absolute(__image3dSize->roi());
    const auto [amin, amax] = basic::liang_barskey(&roi, &cryPoint1, &cryPoint2);
    if (amin >= amax)
      return 0;
    return (roi.x - roi.y).l2() / __image3dSize->voxelSize.l2() * m_sampleRatio + 3;
  }

  __PNI_CUDA_MACRO__ bool setInfo(
      const basic::Event<T> &__projectionEvent, const basic::Image3DGeometry *__image3dSize,
      const basic::Vec2<p3df> *__roi = nullptr) {
    using basic::sgn;
    using basic::smallFloat;
    const p3df cryPoint1 = __projectionEvent.crystal1.geometry->position;
    const p3df cryPoint2 = __projectionEvent.crystal2.geometry->position;
    const auto cry2_cry1 = cryPoint2 - cryPoint1;

    const auto roi = __roi ? basic::cube_and(*__roi, basic::cube_absolute(__image3dSize->roi()))
                           : basic::cube_absolute(__image3dSize->roi());
    const auto [amin, amax] = basic::liang_barskey(&roi, &cryPoint1, &cryPoint2);
    if (amax <= amin)
      return false;

    const float distance_cry1_cry2 = cry2_cry1.l2();
    m_distance_p1_p2 = distance_cry1_cry2 * (amax - amin);

    m_p1 = cryPoint1 + cry2_cry1 * amin;
    m_p2 = cryPoint1 + cry2_cry1 * amax;
    m_p2_p1 = m_p2 - m_p1;

    const auto sampleNumBase = (roi.x - roi.y).l2() / __image3dSize->voxelSize.l2();
    m_sampleNum = static_cast<int>(sampleNumBase * m_sampleRatio + 3);

    constexpr auto SAMPLE_PRECISION = 1023;
    static_assert(SAMPLE_PRECISION > 0, "SAMPLE_PRECISION must be greater than 0");
    m_sampleOffset = ((__projectionEvent.eventIndex % SAMPLE_PRECISION) + 0.5) / T(SAMPLE_PRECISION);
    m_sampleIndex = 0;
    return true;
  }
  __PNI_CUDA_MACRO__ void reset() { m_sampleIndex = 0; }
  __PNI_CUDA_MACRO__ p3df next() {
    const auto nextPoint = m_p1 + m_p2_p1 * (m_sampleIndex + m_sampleOffset) / m_sampleNum;
    m_sampleIndex++;
    return nextPoint;
  }

  __PNI_CUDA_MACRO__ T getlastStepSize() const { return m_distance_p1_p2 / m_sampleNum; }
  __PNI_CUDA_MACRO__ bool isEnd() const { return !(m_sampleIndex < m_sampleNum); }

  __PNI_CUDA_MACRO__ void setSampleRatio(
      T ratio) {
    m_sampleRatio = ratio;
  }

private:
  p3df m_p1, m_p2;
  p3df m_p2_p1; // 方向向量
  T m_sampleOffset;
  int m_sampleIndex;
  int m_sampleNum;    // 采样点总数
  T m_distance_p1_p2; // p1到p2的距离
  T m_sampleRatio{1.f};
};

template <FloatingPoint_c T>
struct SamplingUniformWithTOF {
  __PNI_CUDA_MACRO__ bool setInfo(
      const basic::Event<T> &__projectionEvent, const basic::Image3DGeometry *__image3dSize,
      const basic::Vec2<p3df> *__roi = nullptr) {
    using basic::sgn;
    using basic::smallFloat;
    const p3df cryPoint1 = __projectionEvent.crystal1.geometry->position;
    const p3df cryPoint2 = __projectionEvent.crystal2.geometry->position;
    using fmath = basic::FMath<T>;

    if (!__projectionEvent.crystal1.tof_deviation || !__projectionEvent.crystal2.tof_deviation)
      return false; // 如果没有TOF信息，则不进行采样

    const auto cry2_cry1 = cryPoint2 - cryPoint1;

    const auto roi = __roi ? basic::cube_and(*__roi, basic::cube_absolute(__image3dSize->roi()))
                           : basic::cube_absolute(__image3dSize->roi());
    const auto [amin, amax] = basic::liang_barskey(&roi, &cryPoint1, &cryPoint2);
    if (amax <= amin)
      return false;
    alphaMin = amin;
    alphaMax = amax;

    m_distance_cry1_cry2 = cry2_cry1.l2();
    m_distance_p1_p2 = m_distance_cry1_cry2 * (amax - amin);

    m_p1 = cryPoint1 + cry2_cry1 * amin;
    m_p2 = cryPoint1 + cry2_cry1 * amax;
    m_p2_p1 = m_p2 - m_p1;

    // tofCenter
    m_tofCenter_mm = (*__projectionEvent.time1_2 + basic::value_or(__projectionEvent.crystal1.tof_mean, int16_t(0)) -
                      basic::value_or(__projectionEvent.crystal2.tof_mean, int16_t(0))) *
                         misc::speed_of_light_ps / 2 +
                     m_distance_cry1_cry2 / 2;

    //(0,length_cyry1_cry2)
    m_gaussianStandardDeviation_mm =
        fmath::fsqrt(fmath::fpow(static_cast<T>(*__projectionEvent.crystal1.tof_deviation), 2) +
                     fmath::fpow(static_cast<T>(*__projectionEvent.crystal2.tof_deviation), 2)) *
        misc::speed_of_light_ps;
    constexpr auto SAMPLE_NUM_FAKE_RANDOM = 9;
    const auto sampleNumBase = (roi.x - roi.y).l2() / __image3dSize->voxelSize.l2();
    m_sampleNum =
        static_cast<int>(sampleNumBase * m_sampleRatio + (__projectionEvent.eventIndex % SAMPLE_NUM_FAKE_RANDOM) + 1);

    m_sampleNum = 100;

    constexpr auto SAMPLE_PRECISION = 1023;
    static_assert(SAMPLE_PRECISION > 0, "SAMPLE_PRECISION must be greater than 0");
    m_sampleOffset = ((__projectionEvent.eventIndex % SAMPLE_PRECISION) + 0.5) / T(SAMPLE_PRECISION);
    m_sampleIndex = 0;
    return true;
  }
  __PNI_CUDA_MACRO__ void reset() { m_sampleIndex = 0; }
  __PNI_CUDA_MACRO__ p3df next() {
    const auto nextPoint = m_p1 + m_p2_p1 * (m_sampleIndex + m_sampleOffset) / m_sampleNum;
    const float guassianX1_alpha = alphaMin + (alphaMax - alphaMin) * m_sampleIndex / m_sampleNum;
    const float guassianX1_mm = guassianX1_alpha * m_distance_cry1_cry2;
    const float guassianX2_alpha = alphaMin + (alphaMax - alphaMin) * (m_sampleIndex + 1) / m_sampleNum;
    const float guassianX2_mm = guassianX2_alpha * m_distance_cry1_cry2;

    m_gaussianIntegral = basic::FMath<float>::gauss_integral(guassianX1_mm, guassianX2_mm, m_tofCenter_mm,
                                                             m_gaussianStandardDeviation_mm); // Integral from
    m_sampleIndex++;
    return nextPoint;
  }

  __PNI_CUDA_MACRO__ T getlastStepSize() const { return m_gaussianIntegral; }
  __PNI_CUDA_MACRO__ bool isEnd() const { return !(m_sampleIndex < m_sampleNum); }

  __PNI_CUDA_MACRO__ void setSampleRatio(
      T ratio) {
    m_sampleRatio = ratio;
  }

private:
  p3df m_p1, m_p2;
  p3df m_p2_p1; // 方向向量
  T m_sampleOffset;
  float alphaMin;
  float alphaMax;
  int m_sampleIndex;
  int m_sampleNum;        // 采样点总数
  T m_distance_cry1_cry2; // cry1到cry2的距离
  T m_distance_p1_p2;     // p1到p2的距离
  T m_sampleRatio{1.f};
  T m_gaussianIntegral;
  float m_tofCenter_mm;
  float m_gaussianStandardDeviation_mm;
};

// template < FloatingPoint_c
// struct SamplingGuassianImportance
// {
// public:
//     T tofSamplingRatio;

// public:
//     __PNI_CUDA_MACRO__ bool setInfo(const basic::Event<T> &__projectionEvent,
//                                     const basic::Image3DGeometry *__image3dSize)
//     {
//         basic::Point3D<float> cryPoint1 = *__projectionEvent.position1;
//         basic::Point3D<float> cryPoint2 = *__projectionEvent.position2;
//         const auto cry2_cry1 = cryPoint2 - cryPoint1;

//         const auto [amin, amax] = basic::liang_barskey(__image3dSize, &cryPoint1,
//         &cryPoint2); if (amax < amin)
//             return false;

//         const float distance_cry1_cry2 = cry2_cry1.l2();

//         // 缓存常用值
//         p1 = cryPoint1 + cry2_cry1 * amin;
//         p2 = cryPoint1 + cry2_cry1 * amax;
//         p2_p1 = p2 - p1;
//         image3dSize = __image3dSize;
//         const float a_of_center_of_TOF = *__projectionEvent.time1_2 *
//         misc::speed_of_light_ps / 2.f / distance_cry1_cry2 + 0.5f; center_of_TOF =
//         basic::linearInterpolation(a_of_center_of_TOF, amin, 0.f, amax, 1.f); //
//         假设时间是从0到1的范围 CTR_hwhm = 114514; // TODO: 这个字段待填写

//         const auto distance = p2_p1.l2();
//         sigma = basic::HWHM2Sigma(CTR_hwhm * misc::speed_of_light_ps / distance); //
//         sigma与参数方程的范围对齐 T sigmaBegin = std::max(-center_of_TOF / sigma,
//         -skipSigma); T sigmaEnd = std::min((1 - center_of_TOF) / sigma, skipSigma);
//         // TODO: 没写完
//         samplingDepth = 1919810;
//         stepSize = nullptr;
//         sampleBegin = nullptr;
//         sampleEnd = nullptr;

//         return true;
//     }

// private:
//     T *stepSize;
//     int samplingDepth; // 采样深度，值为N代表一共有2^N个采样点
//     T *sampleBegin;
//     T *sampleEnd;
//     T skipSigma;
//     T sigma;
//     // 预计算的常量
//     basic::Point3D<float> p1, p2;
//     basic::Point3D<float> p2_p1; // 方向向量
//     const basic::Image3DGeometry *image3dSize;
//     float center_of_TOF; // TOF的中心值，参数是从p1到p2的线段参数
//     uint16_t CTR_hwhm;   // 符合时间分辨率(半宽半高)，单位为ps
// };
} // namespace openpni::math
