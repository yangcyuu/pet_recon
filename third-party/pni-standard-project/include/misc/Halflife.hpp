#pragma once
#include <cmath>
namespace openpni::basic {
struct Dose {
  double dose;
  double time;
  double halfLife;

  Dose operator>>(
      double __time) const {
    return {dose * std::exp2(-__time / halfLife), time + __time, halfLife};
  }
  Dose operator<<(
      double __time) const {
    return *this >> -__time;
  }
  double operator>(
      double __time) const {
    return dose * std::exp2(-__time / halfLife);
  }
  double operator<(
      double __time) const {
    return *this > -__time;
  }
  Dose operator+(
      const Dose &other) const {
    if (time == other.time)
      return {dose + other.dose, time, halfLife};
    else {
      const auto thisDoseAtOtherTime = *this >> (other.time - time);
      return {thisDoseAtOtherTime.dose + other.dose, other.time, halfLife};
    }
  }
  Dose operator/(
      double __ratio) const {
    return {dose / __ratio, time, halfLife};
  }
};
} // namespace openpni::basic
