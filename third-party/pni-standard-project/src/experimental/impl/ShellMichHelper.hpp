#pragma once
#include "Projection.h"
#include "include/Exceptions.hpp"
#include "include/experimental/core/Image.hpp"
#include "include/experimental/core/Mich.hpp"
#include "include/experimental/core/Random.hpp"
#include "include/experimental/node/MichCrystal.hpp"
#include "include/experimental/tools/Parallel.hpp"
#include "include/experimental/tools/UtilFunctions.hpp"
#ifndef MichInfoHub
#define MichInfoHub(m) core::MichInfoHub::create(m)
#endif
#ifndef IndexConverter
#define IndexConverter(m) core::IndexConverter::create(m)
#endif
#ifndef RangeGenerator
#define RangeGenerator(m) core::RangeGenerator::create(m)
#endif
namespace openpni::experimental::node::impl {
class ShellMichHelper {
public:
  ShellMichHelper(
      core::MichDefine __mich)
      : m_mich(__mich)
      , m_michCrystal(__mich) {}

public:
  void setShellSize(
      float innerRadius, float outerRadius, float axialLength) {
    m_innerRadius = innerRadius;
    m_outerRadius = outerRadius;
    m_axialLength = axialLength;
    m_maxValue = std::nullopt;
    m_shellImage.clear();
    m_tempSlice.clear();
  }
  void setGrids(
      core::Grids<3, float> const &grids) {
    m_grids = grids;
    m_shellImage.clear();
    m_tempSlice.clear();
  }
  float const *getOneMichSlice(
      int sliceIndex) {
    if (sliceIndex < 0 || sliceIndex >= MichInfoHub(m_mich).getSliceNum())
      throw exceptions::algorithm_unexpected_condition("sliceIndex out of range");
    if (m_shellImage.empty())
      genShellImage();
    if (m_tempSlice.size() != MichInfoHub(m_mich).getBinNum() * MichInfoHub(m_mich).getViewNum())
      m_tempSlice.resize(MichInfoHub(m_mich).getBinNum() * MichInfoHub(m_mich).getViewNum());
    auto lorBegin = sliceIndex * m_tempSlice.size();
    auto lorEnd = lorBegin + m_tempSlice.size();
    auto lors = tools::fill_vector(lorBegin, lorEnd);
    auto *crystalGeoms = m_michCrystal.getHCrystalsBatch(std::span<std::size_t const>(lors));
    tools::parallel_for_each(lors.size(), [&](std::size_t index) {
      const auto cry1 = crystalGeoms[index * 2].O;
      const auto cry2 = crystalGeoms[index * 2 + 1].O;
      m_tempSlice[index] = instant_path_integral(
          core::instant_random_float(index), core::TensorDataInput<float, 3>{m_grids, m_shellImage.data()}, cry1, cry2);
    });
    return m_tempSlice.data();
  }
  float max_value() {
    if (m_maxValue.has_value())
      return m_maxValue.value();
    else
      m_maxValue = cal_max_value();
    return m_maxValue.value();
  }

private:
  void genShellImage() {
    m_shellImage.resize(m_grids.size.totalSize());
    for (const auto &index : m_grids.index_span()) {
      const auto point = m_grids.voxel_center(index);
      auto [x, y, z] = point;
      if (algorithms::l2(point.right_shrink<1>()) >= m_innerRadius &&
          algorithms::l2(point.right_shrink<1>()) <= m_outerRadius && std::abs(point[2]) <= m_axialLength / 2.f) {
        m_shellImage[m_grids.size[index]] = 1.f;
      } else {
        m_shellImage[m_grids.size[index]] = 0.f;
      }
    }
    // if (m_gaussianConvolution)
    //   m_gaussianConvolution->conv(core::TensorDataIO<float, 3>{m_grids, m_shellImage.data(), m_shellImage.data()});
    m_maxValue = std::nullopt;
  }
  float cal_max_value() {
    m_maxValue = std::nullopt;
    for (const auto sliceIndex : std::views::iota(0u, MichInfoHub(m_mich).getSliceNum()))
      m_maxValue = std::max(m_maxValue.value_or(0), *std::max_element(m_shellImage.begin(), m_shellImage.end()));
    return *m_maxValue;
  }

private:
  core::MichDefine m_mich;
  core::Grids<3, float> m_grids;
  std::vector<float> m_shellImage;
  std::vector<float> m_tempSlice;
  std::vector<core::MichStandardEvent> m_tempMichEvents;
  std::optional<float> m_maxValue;

  float m_innerRadius = 0.f;
  float m_outerRadius = 0.f;
  float m_axialLength = 0.f;

  MichCrystal m_michCrystal;
  // std::unique_ptr<GaussianConvolutionCPU<float, 3>> m_gaussianConvolution;
};

} // namespace openpni::experimental::node::impl
