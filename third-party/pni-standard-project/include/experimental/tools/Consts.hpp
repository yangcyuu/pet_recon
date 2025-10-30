#pragma once
#include "../core/BasicMath.hpp"
namespace openpni::experimental::tools {
inline constexpr float flt_max = 3.402823466e+38F;
inline constexpr float speed_of_light = 299792458.0e3f;                      // 光速，单位为毫米/秒
inline constexpr float speed_of_light_ps = 299792458.0e-9f;                  // 光速，单位为毫米/皮秒
inline constexpr float guassian_0_0p5_sigma_integral = 0.19146246127401309f; // 高斯分布在[0, 0.5]范围内的积分值
inline constexpr float guassian_0_1_sigma_integral = 0.34134474606854293f;   // 高斯分布在[0, 1]范围内的积分值
inline constexpr float guassian_0_1p5_sigma_integral = 0.43319279873114191f; // 高斯分布在[0, 1.5]范围内的积分值
inline constexpr float guassian_0_2_sigma_integral = 0.47724986805182079f;   // 高斯分布在[0, 2]范围内的积分值
inline constexpr float guassian_0_2p5_sigma_integral = 0.49379033467422384f; // 高斯分布在[0, 2.5]范围内的积分值
inline constexpr float guassian_0_3_sigma_integral = 0.4986501019683699f;    // 高斯分布在[0, 3]范围内的积分值
inline constexpr float pi = 3.14159265358979323846f;
inline constexpr float pi_2 = 1.57079632679489661923f;
inline constexpr float sqrt3 = 1.73205080756887729353f; // sqrt(3)
template <FloatingPoint_c T>
inline constexpr T sqrt2ln2{std::sqrt(T(2) * std::log(T(2)))}; // sqrt(2*ln(2))
} // namespace openpni::experimental::tools
