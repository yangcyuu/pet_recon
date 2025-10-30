#pragma once
#include <atomic>

#include "../core/Image.hpp"
namespace openpni::experimental::core {
template <typename T>
__PNI_CUDA_MACRO__ void atomic_add(
    T &ref, T value) {
#ifdef __CUDA_ARCH__
  atomicAdd(&ref, value);
#else
  std::atomic_ref(ref) += value;
#endif
}
} // namespace openpni::experimental::core
namespace openpni::experimental::core {
template <FloatingPoint_c F, int D>
class InterpolationNearestGetter {
public:
  __PNI_CUDA_MACRO__ InterpolationNearestGetter(
      core::TensorDataInput<F, D> __image) noexcept
      : m_image(__image) {}
  __PNI_CUDA_MACRO__ ~InterpolationNearestGetter(){};

public:
  template <FloatingPoint_c FF = F>
  __PNI_CUDA_MACRO__ F get(
      const core::Vector<FF, D> &point) const noexcept {
    auto index = m_image.grid.find_index_from_float(point.template to<F>());
    return m_image.grid.in(index) ? m_image.ptr[m_image.grid.at(index)] : 0;
  }

private:
  core::TensorDataInput<F, D> m_image;
};
template <FloatingPoint_c F, int D>
class InterpolationNearestSetter {
public:
  __PNI_CUDA_MACRO__
  InterpolationNearestSetter(
      core::TensorDataOutput<F, D> __image) noexcept
      : m_image(__image) {}
  __PNI_CUDA_MACRO__ ~InterpolationNearestSetter(){};

  template <FloatingPoint_c, int>
  friend class InterpolationNearest;

public:
  template <FloatingPoint_c FF = F>
  __PNI_CUDA_MACRO__ void set(
      const core::Vector<FF, D> &point, const auto &value) const noexcept {
    auto index = m_image.grid.find_index_from_float(point);
    if (m_image.grid.in(index))
      m_image.ptr[m_image.grid.at(index)] = value;
  }
  template <FloatingPoint_c FF = F>
  __PNI_CUDA_MACRO__ void add(
      const core::Vector<FF, D> &point, const auto &value) const noexcept {
    auto index = m_image.grid.find_index_from_float(point);
    if (m_image.grid.in(index))
      atomic_add(m_image.ptr[m_image.grid.at(index)], value);
  }

private:
  core::TensorDataOutput<F, D> m_image;
};

template <FloatingPoint_c F, int D>
class InterpolationNearest {
public:
  __PNI_CUDA_MACRO__
  InterpolationNearest(
      core::TensorDataIO<F, D> __image) noexcept
      : m_getter(__image.input_branch())
      , m_setter(__image.output_branch()) {}
  __PNI_CUDA_MACRO__
  ~InterpolationNearest() {};

public:
  template <FloatingPoint_c FF = F>
  __PNI_CUDA_MACRO__ F get(
      const core::Vector<FF, D> &point) const noexcept {
    return m_getter.template get<FF>(point);
  }

  template <FloatingPoint_c FF = F, typename ValueType>
  __PNI_CUDA_MACRO__ void set(
      const core::Vector<FF, D> &point, const ValueType &value) const noexcept {
    m_setter.template set<FF>(point, value);
  }

  template <FloatingPoint_c FF = F, typename ValueType>
  __PNI_CUDA_MACRO__ void add(
      const core::Vector<FF, D> &point, const ValueType &value) const noexcept {
    m_setter.template add<FF>(point, value);
  }
  __PNI_CUDA_MACRO__
  auto &getter() { return m_getter; }
  __PNI_CUDA_MACRO__
  auto &setter() { return m_setter; }

private:
  InterpolationNearestGetter<F, D> m_getter;
  InterpolationNearestSetter<F, D> m_setter;
};

} // namespace openpni::experimental::core
namespace openpni::experimental::core {
template <FloatingPoint_c F, int D>
class InterpolationLinearGetter {
private:
  constexpr inline int static _power_2 = 1 << D;
  using TempValues = core::Vector<F, _power_2>;

public:
  __PNI_CUDA_MACRO__
  InterpolationLinearGetter(
      core::TensorDataInput<F, D> __image) noexcept
      : m_image(__image) {}
  __PNI_CUDA_MACRO__
  ~InterpolationLinearGetter() {};

public:
  template <FloatingPoint_c FF = F>
  __PNI_CUDA_MACRO__ F get(
      const core::Vector<FF, D> &point) const noexcept {
    constexpr core::Vector<F, D> bias = core::Vector<F, D>::create(0.5);
    const auto biased_point = point.template to<F>() + bias;
    const auto index = m_image.grid.find_index_from_float(biased_point);
    if (m_image.grid.in(index) == false)
      return 0;

    core::Vector<F, D> _0 = (biased_point.template to<F>()) - index.template to<F>();
    core::Vector<F, D> _1 = core::Vector<F, D>::create(1) - _0;
    TempValues values = values(index);
    TempValues coefs = coefs(_0, _1);
    const auto coefs_sum = coefs.l1();
    if (coefs_sum < 1e-8) // The point is out of the image
      return 0;
    return values.dot(coefs) / coefs_sum;
  }

private:
  template <int d>
  __PNI_CUDA_MACRO__ static constexpr int64_t power_2() {
    if constexpr (d == 0)
      return 1;
    else
      return 2 * power_2<d - 1>();
  }
  __PNI_CUDA_MACRO__
  TempValues values(
      const core::Vector<int64_t, D> &index) const noexcept {
    TempValues result = TempValues::create(0);
    core::MDSpan<D> span01 = core::MDSpan<D>::create(core::Vector<int64_t, D>::create(2));
    for (const auto &delta01 : span01) {
      auto index_delta = index + delta01;
      if (m_image.grid.in(index_delta) == false)
        result[delta01] = 0;
      else
        result[delta01] = m_image.ptr[m_image.grid.at(index_delta)];
    }
    return result;
  }
  __PNI_CUDA_MACRO__
  TempValues coefs(
      const core::Vector<F, D> &_0, const core::Vector<F, D> &_1) const noexcept {
    TempValues result = TempValues::create(1);
    coef_distribute<0>(_0, _1, result);
    return result;
  }

  template <int d>
  __PNI_CUDA_MACRO__ void coef_distribute(
      const core::Vector<F, D> &_0, const core::Vector<F, D> &_1, TempValues &coefs) const noexcept {
    constexpr int bit_mask = 1 << d;
    for (int i = 0; i < power_2<D>(); i++) {
      if ((i & bit_mask) == 0)
        coefs[i] *= _1[d];
      else
        coefs[i] *= _0[d];
    }
    if constexpr (d + 1 < D)
      coef_distribute<d + 1>(_0, _1, coefs);
  }

private:
  core::TensorDataInput<F, D> m_image;
};
} // namespace openpni::experimental::core